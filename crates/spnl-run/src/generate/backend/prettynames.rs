//! Pretty name mapping for local models
//!
//! This module provides a user-friendly naming scheme for models, similar to Ollama.
//! It maps short, memorable names like "llama3.1:8b" to their full HuggingFace model paths.
//! All lookups delegate to the mistralrs backend for actual inference.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::{
    SpnlResult,
    generate::GenerateOptions,
    ir::{Map, Repeat},
};

/// Static lookup table mapping pretty names to HuggingFace model names
/// Format: "{provider}{version}:{size}" -> "unsloth/{Model}-GGUF"
static MODEL_LOOKUP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();

/// Initialize the model lookup table
fn init_model_lookup() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();

    // Llama models
    m.insert("llama3.1:8b", "unsloth/Llama-3.1-8B-Instruct-GGUF");
    m.insert("llama3.1:70b", "unsloth/Llama-3.1-70B-Instruct-GGUF");
    m.insert("llama3.2:1b", "unsloth/Llama-3.2-1B-Instruct-GGUF");
    m.insert("llama3.2:3b", "unsloth/Llama-3.2-3B-Instruct-GGUF");
    m.insert("llama3.3:70b", "unsloth/Llama-3.3-70B-Instruct-GGUF");

    // Gemma models
    m.insert("gemma2:3b", "unsloth/gemma-2-it-GGUF");
    m.insert("gemma3:270m", "unsloth/gemma-3-270m-it-GGUF");
    m.insert("gemma3:1b", "unsloth/gemma-3-1b-it-GGUF");
    m.insert("gemma3:4b", "unsloth/gemma-3-4b-it-GGUF");
    m.insert("gemma3:12b", "unsloth/gemma-3-12b-it-GGUF");
    m.insert("gemma3:27b", "unsloth/gemma-3-27b-it-GGUF");

    // Phi models
    m.insert("phi3:3b", "unsloth/Phi-3-mini-4k-instruct-GGUF");
    m.insert("phi3.5:3b", "unsloth/Phi-3.5-mini-instruct-GGUF");
    m.insert("phi4:14b", "unsloth/Phi-4-GGUF");

    // Qwen models
    //m.insert("qwen2:0.5b", "unsloth/Qwen2-0.5B-Instruct-GGUF");
    //m.insert("qwen2:1.5b", "unsloth/Qwen2-1.5B-Instruct-GGUF");
    //m.insert("qwen2:7b", "unsloth/Qwen2-7B-Instruct-GGUF");
    m.insert("qwen2.5:0.5b", "unsloth/Qwen2.5-Coder-0.5B-Instruct-GGUF");
    m.insert("qwen2.5:1.5b", "unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF");
    m.insert("qwen2.5:3b", "unsloth/Qwen2.5-Coder-3B-Instruct-GGUF");
    m.insert("qwen2.5:7b", "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF");
    m.insert("qwen2.5:14b", "unsloth/Qwen2.5-Coder-14B-Instruct-GGUF");
    m.insert("qwen2.5:32b", "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF");
    m.insert("qwen2.5:72b", "unsloth/Qwen2.5-Coder-72B-Instruct-GGUF");

    m
}

/// Look up a pretty name and return the full HuggingFace model path
/// Returns None if the pretty name is not found
pub fn lookup_model(pretty_name: &str) -> Option<&'static str> {
    MODEL_LOOKUP
        .get_or_init(init_model_lookup)
        .get(pretty_name)
        .copied()
}

/// Generate completions using a pretty model name
/// Looks up the model and delegates to mistralrs backend
pub async fn generate_completion(
    spec: &Map,
    mp: Option<&indicatif::MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    // Look up the pretty name
    let model_name = &spec.metadata.model;
    let full_model_name =
        lookup_model(model_name).ok_or_else(|| anyhow::anyhow!("Unknown model: {}", model_name))?;

    // Create a new spec with the full model name and delegate to mistralrs
    let new_spec = spec.with_model(full_model_name)?;
    super::mistralrs::generate_completion(new_spec, mp, options).await
}

/// Generate chat completions using a pretty model name
/// Looks up the model and delegates to mistralrs backend
pub async fn generate_chat(
    spec: &Repeat,
    mp: Option<&indicatif::MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    // Look up the pretty name
    let model_name = &spec.generate.metadata.model;
    let full_model_name =
        lookup_model(model_name).ok_or_else(|| anyhow::anyhow!("Unknown model: {}", model_name))?;

    // Create a new spec with the full model name and delegate to mistralrs
    let new_spec = spec.with_model(full_model_name)?;
    super::mistralrs::generate_chat(new_spec, mp, options).await
}

// Made with Bob

/// Get the HuggingFace cache directory
fn get_hf_cache_dir() -> PathBuf {
    // Check HF_HOME first (official HuggingFace env var)
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }

    // Fall back to default: ~/.cache/huggingface/hub
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .expect("Could not determine home directory");

    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// Check if a model is in the HuggingFace cache
/// Returns true if the model directory exists in the cache
pub fn is_model_cached(hf_model_name: &str) -> bool {
    let cache_dir = get_hf_cache_dir();

    // HuggingFace stores models with "models--" prefix and "/" replaced with "--"
    let model_dir_name = format!("models--{}", hf_model_name.replace('/', "--"));
    let model_path = cache_dir.join(&model_dir_name);

    model_path.exists()
}

/// Get all available pretty names with their HuggingFace model paths
/// Returns a vector of (pretty_name, hf_model_name, is_cached) tuples
pub fn list_all_models() -> Vec<(&'static str, &'static str, bool)> {
    let lookup = MODEL_LOOKUP.get_or_init(init_model_lookup);
    let mut models: Vec<_> = lookup
        .iter()
        .map(|(pretty_name, hf_name)| (*pretty_name, *hf_name, is_model_cached(hf_name)))
        .collect();

    // Sort by pretty name for consistent output
    models.sort_by(|a, b| a.0.cmp(b.0));
    models
}
