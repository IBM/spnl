//! Model loading and caching for mistral.rs backend

use mistralrs::{
    Device, GgufModelBuilder, IsqType, Model, PagedAttentionMetaBuilder, TextModelBuilder,
    paged_attn_supported,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Get the HuggingFace cache directory
/// This matches the cache used by hf_hub crate (used by candle backend)
/// Returns the path to the hub directory where models are stored
fn get_hf_cache_dir() -> PathBuf {
    // Check HF_HOME first (official HuggingFace env var)
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }

    // Fall back to default: ~/.cache/huggingface/hub
    // This is where hf_hub stores models with the models-- prefix
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .expect("Could not determine home directory");

    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// Get PagedAttention configuration from environment variables
/// Returns None if PagedAttention is disabled or not supported
fn get_paged_attn_config()
-> Option<impl FnOnce() -> anyhow::Result<mistralrs::PagedAttentionConfig>> {
    // Check if PagedAttention is supported on this platform
    if !paged_attn_supported() {
        eprintln!("PagedAttention not supported on this platform");
        return None;
    }

    // Check if explicitly disabled via environment variable
    let enabled = std::env::var("MISTRALRS_PAGED_ATTN")
        .unwrap_or_else(|_| "true".to_string())
        .to_lowercase();

    if enabled == "false" || enabled == "0" {
        eprintln!("PagedAttention disabled via MISTRALRS_PAGED_ATTN");
        return None;
    }

    // Get block size from environment or use default
    let block_size = std::env::var("MISTRALRS_PAGED_ATTN_BLOCK_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);

    eprintln!("Enabling PagedAttention with block_size={}", block_size);

    Some(move || {
        PagedAttentionMetaBuilder::default()
            .with_block_size(block_size)
            .build()
    })
}

/// Get prefix cache size from environment variables
/// Returns None to disable prefix caching, or Some(n) to cache n sequences
fn get_prefix_cache_n() -> Option<usize> {
    match std::env::var("MISTRALRS_PREFIX_CACHE_N") {
        Ok(val) => {
            if val.to_lowercase() == "false" || val == "0" {
                eprintln!("Prefix caching disabled via MISTRALRS_PREFIX_CACHE_N");
                None
            } else {
                match val.parse::<usize>() {
                    Ok(n) => {
                        eprintln!("Prefix caching enabled with n={}", n);
                        Some(n)
                    }
                    Err(_) => {
                        eprintln!("Invalid MISTRALRS_PREFIX_CACHE_N value, using default (16)");
                        Some(16)
                    }
                }
            }
        }
        Err(_) => {
            // Default: enable with 16 sequences
            Some(16)
        }
    }
}

/// Parse ISQ (In-Situ Quantization) type from string
/// Returns None if quantization is disabled or invalid
fn parse_isq_type(isq_str: &str) -> anyhow::Result<IsqType> {
    // Normalize the input: uppercase and remove common suffixes
    let normalized = isq_str
        .to_uppercase()
        .replace("_M", "")
        .replace("_S", "")
        .replace("_L", "");

    match normalized.as_str() {
        "Q4K" | "Q4_K" => Ok(IsqType::Q4K),
        "Q4_0" | "Q40" => Ok(IsqType::Q4_0),
        "Q4_1" | "Q41" => Ok(IsqType::Q4_1),
        "Q5K" | "Q5_K" => Ok(IsqType::Q5K),
        "Q5_0" | "Q50" => Ok(IsqType::Q5_0),
        "Q5_1" | "Q51" => Ok(IsqType::Q5_1),
        "Q8_0" | "Q80" | "Q8" => Ok(IsqType::Q8_0),
        "Q8_1" | "Q81" => Ok(IsqType::Q8_1),
        "Q2K" | "Q2_K" => Ok(IsqType::Q2K),
        "Q3K" | "Q3_K" => Ok(IsqType::Q3K),
        "Q6K" | "Q6_K" => Ok(IsqType::Q6K),
        _ => Err(anyhow::anyhow!(
            "Invalid ISQ type: {}. Valid options: Q2K, Q3K, Q4K (or Q4_K_M), Q4_0, Q4_1, Q5K (or Q5_K_M), Q5_0, Q5_1, Q6K, Q8_0, Q8_1",
            isq_str
        )),
    }
}

/// Get ISQ configuration from environment variables
/// Returns None if quantization is disabled
fn get_isq_type() -> Option<IsqType> {
    match std::env::var("MISTRALRS_ISQ") {
        Ok(val) => {
            if val.to_lowercase() == "false" || val.to_lowercase() == "none" {
                eprintln!("In-situ quantization disabled via MISTRALRS_ISQ");
                None
            } else {
                match parse_isq_type(&val) {
                    Ok(isq_type) => {
                        eprintln!("Enabling in-situ quantization: {:?}", isq_type);
                        Some(isq_type)
                    }
                    Err(e) => {
                        eprintln!("Warning: {}", e);
                        None
                    }
                }
            }
        }
        Err(_) => {
            // Default: no quantization for non-GGUF models
            None
        }
    }
}

/// Model pool for caching loaded models
pub struct ModelPool {
    models: Arc<RwLock<HashMap<String, Arc<Model>>>>,
}

impl ModelPool {
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or load a model
    pub async fn get_or_load(&self, model_name: &str) -> anyhow::Result<Arc<Model>> {
        // Check if model is already loaded
        {
            let models = self.models.read().await;
            if let Some(model) = models.get(model_name) {
                return Ok(Arc::clone(model));
            }
        }

        // Model not in cache, load it
        let model = self.load_model(model_name).await?;

        // Cache the loaded model
        {
            let mut models = self.models.write().await;
            models.insert(model_name.to_string(), Arc::clone(&model));
        }

        Ok(model)
    }

    /// Select GGUF files from a HuggingFace repository
    /// First checks local cache, then queries HF API if needed
    async fn select_gguf_files(&self, model_name: &str) -> anyhow::Result<Vec<String>> {
        // Extract base name from model_id (e.g., "Qwen/Qwen3-1.7B-GGUF" -> "Qwen3-1.7B")
        let base_name = model_name
            .split('/')
            .next_back()
            .unwrap_or(model_name)
            .replace("-GGUF", "")
            .replace("-gguf", "");

        // Priority list of quantization formats to try
        // Support multiple naming conventions:
        // 1. "model-name-Q4_K_M.gguf" (hyphen separator)
        // 2. "model-name.Q4_K_M.gguf" (dot separator - QuantFactory style)
        let priority_formats = vec![
            // Q8_0 (higher quality)
            format!("{}.Q8_0.gguf", base_name),
            format!("{}-Q8_0.gguf", base_name),
            format!("{}-q8_0.gguf", base_name),
            format!("{}.q8_0.gguf", base_name),
            // Q4_K_M (most common, good balance)
            format!("{}.Q4_K_M.gguf", base_name),
            format!("{}-Q4_K_M.gguf", base_name),
            format!("{}-q4_k_m.gguf", base_name),
            format!("{}.q4_k_m.gguf", base_name),
            // Q5_K_M (balanced)
            format!("{}.Q5_K_M.gguf", base_name),
            format!("{}-Q5_K_M.gguf", base_name),
            format!("{}-q5_k_m.gguf", base_name),
            format!("{}.q5_k_m.gguf", base_name),
            // Q4_K_S (smaller, faster)
            format!("{}.Q4_K_S.gguf", base_name),
            format!("{}-Q4_K_S.gguf", base_name),
        ];

        // First, check if any files are already in the local cache
        let hf_cache = get_hf_cache_dir();
        let model_cache_dir = hf_cache.join(format!("models--{}", model_name.replace("/", "--")));

        if model_cache_dir.exists() {
            // Check snapshots directory for cached GGUF files
            if let Ok(entries) = std::fs::read_dir(model_cache_dir.join("snapshots")) {
                for entry in entries.flatten() {
                    if let Ok(snapshot_entries) = std::fs::read_dir(entry.path()) {
                        let cached_files: Vec<String> = snapshot_entries
                            .flatten()
                            .filter_map(|e| {
                                e.file_name()
                                    .to_str()
                                    .filter(|n| n.ends_with(".gguf"))
                                    .map(|n| n.to_string())
                            })
                            .collect();

                        if !cached_files.is_empty() {
                            eprintln!("Found cached GGUF files: {:?}", cached_files);

                            // Return the first priority format that's cached
                            for filename in &priority_formats {
                                if cached_files.contains(filename) {
                                    eprintln!("Using cached GGUF file: {}", filename);
                                    return Ok(vec![filename.clone()]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // If not in cache, query HF API to find which file to download
        eprintln!("Model not in cache, querying HuggingFace API...");
        let url = format!("https://huggingface.co/api/models/{}/tree/main", model_name);
        let response = reqwest::get(&url).await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to fetch model files from HuggingFace: HTTP {}",
                response.status()
            ));
        }

        let files: Vec<serde_json::Value> = response.json().await?;
        let available_files: std::collections::HashSet<String> = files
            .into_iter()
            .filter_map(|f| {
                f.get("path")
                    .and_then(|p| p.as_str())
                    .filter(|p| p.ends_with(".gguf"))
                    .map(|p| p.to_string())
            })
            .collect();

        eprintln!("Available GGUF files in repo: {:?}", available_files);

        // Return the first priority format that exists
        for filename in &priority_formats {
            if available_files.contains(filename) {
                eprintln!("Will download GGUF file: {}", filename);
                return Ok(vec![filename.clone()]);
            }
        }

        // If none of the priority formats exist, return an error with helpful info
        Err(anyhow::anyhow!(
            "No matching GGUF files found in repository {}.\nAvailable files: {:?}\nTried: {:?}",
            model_name,
            available_files,
            priority_formats
        ))
    }

    /// Load a model from HuggingFace using appropriate builder
    async fn load_model(&self, model_name: &str) -> anyhow::Result<Arc<Model>> {
        eprintln!("Loading model: {}", model_name);

        // Check if this is a GGUF model (contains "GGUF" in the name)
        let is_gguf = model_name.to_uppercase().contains("GGUF");

        // Determine device - prefer Metal on macOS, fallback to CPU
        let device = if cfg!(target_os = "macos") {
            match Device::new_metal(0) {
                Ok(metal_device) => {
                    eprintln!("Using Metal GPU acceleration");
                    metal_device
                }
                Err(e) => {
                    eprintln!("Metal not available ({}), falling back to CPU", e);
                    Device::Cpu
                }
            }
        } else {
            Device::Cpu
        };

        // Build the model using the appropriate builder
        let model = if is_gguf {
            eprintln!("Detected GGUF model, using GgufModelBuilder");

            // Get priority list of GGUF files to try
            let gguf_files = self.select_gguf_files(model_name).await?;

            if let Some(first_file) = gguf_files.first() {
                eprintln!("Using GGUF file: {}", first_file);
            }

            // Use GgufModelBuilder for GGUF models
            let mut builder = GgufModelBuilder::new(model_name, gguf_files)
                .with_logging()
                .with_device(device);

            // Apply PagedAttention if available and enabled
            if let Some(paged_config) = get_paged_attn_config() {
                builder = builder.with_paged_attn(paged_config)?;
            }

            // Apply prefix caching configuration
            builder = builder.with_prefix_cache_n(get_prefix_cache_n());

            builder.build().await?
        } else {
            eprintln!("Using TextModelBuilder for standard model");

            // Use TextModelBuilder for normal models
            let mut builder = TextModelBuilder::new(model_name)
                .with_logging()
                // .with_dtype(mistralrs::ModelDType::F32) // for future reference: might be needed for ISQ on metal
                .with_device(device);

            // Apply in-situ quantization if configured
            if let Some(isq_type) = get_isq_type() {
                builder = builder.with_isq(isq_type);
            }

            // Apply PagedAttention if available and enabled
            if let Some(paged_config) = get_paged_attn_config() {
                builder = builder.with_paged_attn(paged_config)?;
            }

            // Apply prefix caching configuration
            builder = builder.with_prefix_cache_n(get_prefix_cache_n());

            builder.build().await?
        };

        eprintln!("Model loaded successfully: {}", model_name);
        Ok(Arc::new(model))
    }
}

impl Default for ModelPool {
    fn default() -> Self {
        Self::new()
    }
}

// Made with Bob
