use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use super::{
    CandleModel, Gemma2GenericConfig, Gemma2ModelWrapper, Gemma3GenericConfig, Gemma3ModelWrapper,
    LlamaGenericConfig, LlamaModelWrapper, Qwen2GenericConfig, Qwen2ModelWrapper,
    Qwen3GenericConfig, Qwen3ModelWrapper, Qwen3MoeGenericConfig, Qwen3MoeModelWrapper,
    quantized_qwen2::QuantizedQwen2ModelWrapper, quantized_qwen3::QuantizedQwen3ModelWrapper,
    quantized_qwen3_moe::QuantizedQwen3MoeModelWrapper,
};
use crate::generate::backend::shared::{GenericConfig, detect_architecture, download_model_files};
/// Download GGUF model from HuggingFace and load it
/// Tries multiple quantization formats in priority order:
/// 1. q4_k_m (best balance of size/quality)
/// 2. Q8_0 (higher quality, larger)
/// 3. q5_k_m (middle ground)
fn download_and_load_gguf(
    model_id: &str,
    device: Device,
) -> anyhow::Result<(Box<dyn CandleModel>, Tokenizer, Device, DType)> {
    use hf_hub::api::sync::Api;

    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    // Base filename without quantization suffix
    // Extract just the model name, removing the organization prefix (e.g., "unsloth/" or "Qwen/")
    let base_name = model_id
        .split('/')
        .next_back()
        .unwrap_or(model_id)
        .replace("-GGUF", "");

    // Try quantization formats in priority order
    // Try both lowercase and uppercase variants back-to-back for each format
    // Some repos use lowercase (q4_k_m), others use uppercase (Q4_K_M)
    let quantization_formats = vec![
        format!("{}-q4_k_m.gguf", base_name), // lowercase q4_k_m
        format!("{}-Q4_K_M.gguf", base_name), // uppercase Q4_K_M
        format!("{}-Q8_0.gguf", base_name),   // uppercase Q8_0
        format!("{}-q8_0.gguf", base_name),   // lowercase q8_0
        format!("{}-q5_k_m.gguf", base_name), // lowercase q5_k_m
        format!("{}-Q5_K_M.gguf", base_name), // uppercase Q5_K_M
    ];

    let mut last_error = None;
    for gguf_filename in quantization_formats {
        // eprintln!("Trying to download: {}", gguf_filename);
        match repo.get(&gguf_filename) {
            Ok(gguf_path) => {
                eprintln!("Successfully downloaded: {}", gguf_filename);
                return load_gguf_model(&gguf_path, model_id, device);
            }
            Err(e) => {
                last_error = Some(e);
                continue;
            }
        }
    }

    Err(anyhow::anyhow!(
        "Could not find any GGUF file for model: {}. Last error: {:?}",
        model_id,
        last_error
    ))
}

/// Load a model from HuggingFace Hub
pub fn load_model(
    model_id: &str,
) -> anyhow::Result<(Box<dyn CandleModel>, Tokenizer, Device, DType)> {
    // Determine device (Metal on macOS if available, otherwise CPU)
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);

    // Check if this is a GGUF model by model ID or by finding .gguf files
    // GGUF models are quantized and don't need dtype selection
    if model_id.contains("-GGUF") || model_id.contains(".gguf") {
        // Try to find existing GGUF file, or download it
        match find_gguf_file(model_id) {
            Ok(gguf_path) => {
                return load_gguf_model(&gguf_path, model_id, device);
            }
            Err(_e) => {
                // GGUF file not found locally, download from HuggingFace
                return download_and_load_gguf(model_id, device);
            }
        }
    } else if let Ok(gguf_path) = find_gguf_file(model_id) {
        return load_gguf_model(&gguf_path, model_id, device);
    }

    // Try F16 first on Metal for better performance (2x faster, 2x less memory)
    // Will automatically fall back to F32 if F16 causes numerical issues
    // Set CANDLE_FORCE_F32=1 to skip F16 and use F32 directly
    let try_f16 = device.is_metal() && std::env::var("CANDLE_FORCE_F32").is_err();

    // Download all necessary files (only once, reused if we need to retry)
    let (tokenizer_path, config_path, filenames) = download_model_files(model_id)?;

    // Gemma3 models have numerical stability issues with F16, force F32
    let force_f32_for_gemma3 =
        model_id.to_lowercase().contains("gemma3") || model_id.to_lowercase().contains("gemma-3");

    // Try loading with F16 first if requested (unless it's Gemma3), fall back to F32 on error
    let dtype = if try_f16 && !force_f32_for_gemma3 {
        DType::F16
    } else {
        DType::F32
    };

    match try_load_model_with_dtype(
        model_id,
        &tokenizer_path,
        &config_path,
        &filenames,
        &device,
        dtype,
    ) {
        Ok(result) => Ok(result),
        Err(e)
            if try_f16
                && !force_f32_for_gemma3
                && (e.to_string().contains("weight")
                    || e.to_string().contains("negative")
                    || e.to_string().contains("not a valid number")) =>
        {
            // F16 failed with numerical error, retry with F32
            eprintln!(
                "F16 failed ({}), retrying with F32 for better numerical stability",
                e
            );
            try_load_model_with_dtype(
                model_id,
                &tokenizer_path,
                &config_path,
                &filenames,
                &device,
                DType::F32,
            )
        }
        Err(e) => Err(e),
    }
}

/// Internal function to load model with a specific dtype
fn try_load_model_with_dtype(
    _model_id: &str,
    tokenizer_path: &std::path::Path,
    config_path: &std::path::Path,
    filenames: &[std::path::PathBuf],
    device: &Device,
    dtype: DType,
) -> anyhow::Result<(Box<dyn CandleModel>, Tokenizer, Device, DType)> {
    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Load and parse config
    let config_str = std::fs::read_to_string(config_path)?;
    let generic_config: GenericConfig = serde_json::from_str(&config_str)?;

    // Detect architecture
    let architecture = detect_architecture(&generic_config)?;

    // Load weights
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(filenames, dtype, device)? };

    // Load the appropriate model based on architecture
    let model: Box<dyn CandleModel> = match architecture.as_str() {
        arch if arch.contains("Gemma2") => {
            let gemma2_config = Gemma2GenericConfig {
                vocab_size: generic_config.vocab_size,
                hidden_size: generic_config.hidden_size,
                intermediate_size: generic_config.intermediate_size,
                num_hidden_layers: generic_config.num_hidden_layers,
                num_attention_heads: generic_config.num_attention_heads,
                num_key_value_heads: generic_config.num_key_value_heads,
                head_dim: generic_config.head_dim,
                max_position_embeddings: generic_config.max_position_embeddings,
                rms_norm_eps: generic_config.rms_norm_eps,
                rope_theta: generic_config.rope_theta,
                attention_bias: generic_config.attention_bias,
                hidden_act: generic_config.hidden_act,
                attn_logit_softcapping: None, // Not in GenericConfig yet
                final_logit_softcapping: None, // Not in GenericConfig yet
                query_pre_attn_scalar: None,  // Not in GenericConfig yet
            };
            let config = gemma2_config.into();
            Box::new(Gemma2ModelWrapper::load(vb, config)?)
        }
        arch if arch.contains("Gemma3") => {
            let gemma3_config = Gemma3GenericConfig {
                vocab_size: generic_config.vocab_size,
                hidden_size: generic_config.hidden_size,
                intermediate_size: generic_config.intermediate_size,
                num_hidden_layers: generic_config.num_hidden_layers,
                num_attention_heads: generic_config.num_attention_heads,
                num_key_value_heads: generic_config.num_key_value_heads,
                head_dim: generic_config.head_dim,
                max_position_embeddings: generic_config.max_position_embeddings,
                rms_norm_eps: generic_config.rms_norm_eps,
                rope_theta: generic_config.rope_theta,
                attention_bias: generic_config.attention_bias,
                hidden_act: generic_config.hidden_act,
                attn_logit_softcapping: None, // Not in GenericConfig yet
                final_logit_softcapping: None, // Not in GenericConfig yet
                query_pre_attn_scalar: None,  // Not in GenericConfig yet
            };
            let config = gemma3_config.into();
            Box::new(Gemma3ModelWrapper::load(vb, config)?)
        }
        arch if arch.contains("Llama") => {
            let llama_config = LlamaGenericConfig {
                hidden_size: generic_config.hidden_size,
                intermediate_size: generic_config.intermediate_size,
                vocab_size: generic_config.vocab_size,
                num_hidden_layers: generic_config.num_hidden_layers,
                num_attention_heads: generic_config.num_attention_heads,
                num_key_value_heads: generic_config.num_key_value_heads,
                rms_norm_eps: generic_config.rms_norm_eps,
                rope_theta: generic_config.rope_theta,
                max_position_embeddings: generic_config.max_position_embeddings,
                bos_token_id: generic_config.bos_token_id,
                eos_token_id: generic_config.eos_token_id,
            };
            let config = llama_config.into();
            Box::new(LlamaModelWrapper::load(vb, config, device.clone(), dtype)?)
        }
        arch if arch.contains("Qwen2") => {
            let qwen2_config = Qwen2GenericConfig {
                vocab_size: generic_config.vocab_size,
                hidden_size: generic_config.hidden_size,
                intermediate_size: generic_config.intermediate_size,
                num_hidden_layers: generic_config.num_hidden_layers,
                num_attention_heads: generic_config.num_attention_heads,
                num_key_value_heads: generic_config.num_key_value_heads,
                max_position_embeddings: generic_config.max_position_embeddings,
                sliding_window: generic_config.sliding_window,
                max_window_layers: generic_config.max_window_layers,
                tie_word_embeddings: generic_config.tie_word_embeddings,
                rope_theta: generic_config.rope_theta,
                rms_norm_eps: generic_config.rms_norm_eps,
                use_sliding_window: generic_config.use_sliding_window,
                hidden_act: generic_config.hidden_act,
            };
            let config = qwen2_config.into();
            Box::new(Qwen2ModelWrapper::load(vb, config)?)
        }
        arch if arch.contains("Qwen3") => {
            // Check if this is a MoE model by looking for num_experts field
            if generic_config.num_experts.is_some() {
                // This is a Qwen3 MoE model (e.g., A3B)
                let qwen3_moe_config = Qwen3MoeGenericConfig {
                    vocab_size: generic_config.vocab_size,
                    hidden_size: generic_config.hidden_size,
                    intermediate_size: generic_config.intermediate_size,
                    num_hidden_layers: generic_config.num_hidden_layers,
                    num_attention_heads: generic_config.num_attention_heads,
                    num_key_value_heads: generic_config.num_key_value_heads,
                    max_position_embeddings: generic_config.max_position_embeddings,
                    sliding_window: generic_config.sliding_window,
                    max_window_layers: generic_config.max_window_layers,
                    tie_word_embeddings: generic_config.tie_word_embeddings,
                    rope_theta: generic_config.rope_theta,
                    rms_norm_eps: generic_config.rms_norm_eps,
                    use_sliding_window: generic_config.use_sliding_window,
                    hidden_act: generic_config.hidden_act,
                    attention_bias: generic_config.attention_bias,
                    head_dim: generic_config.head_dim,
                    num_experts_per_tok: generic_config.num_experts_per_tok,
                    num_experts: generic_config.num_experts,
                    moe_intermediate_size: generic_config.moe_intermediate_size,
                    decoder_sparse_step: generic_config.decoder_sparse_step,
                    norm_topk_prob: generic_config.norm_topk_prob,
                };
                let config = qwen3_moe_config.into();
                Box::new(Qwen3MoeModelWrapper::load(vb, config)?)
            } else {
                // Standard Qwen3 model
                let qwen3_config = Qwen3GenericConfig {
                    vocab_size: generic_config.vocab_size,
                    hidden_size: generic_config.hidden_size,
                    intermediate_size: generic_config.intermediate_size,
                    num_hidden_layers: generic_config.num_hidden_layers,
                    num_attention_heads: generic_config.num_attention_heads,
                    num_key_value_heads: generic_config.num_key_value_heads,
                    max_position_embeddings: generic_config.max_position_embeddings,
                    sliding_window: generic_config.sliding_window,
                    max_window_layers: generic_config.max_window_layers,
                    tie_word_embeddings: generic_config.tie_word_embeddings,
                    rope_theta: generic_config.rope_theta,
                    rms_norm_eps: generic_config.rms_norm_eps,
                    use_sliding_window: generic_config.use_sliding_window,
                    hidden_act: generic_config.hidden_act,
                    attention_bias: generic_config.attention_bias,
                    head_dim: generic_config.head_dim,
                };
                let config = qwen3_config.into();
                Box::new(Qwen3ModelWrapper::load(vb, config)?)
            }
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported architecture: {}. Currently supported: Gemma2, Gemma3, Llama, Qwen2, Qwen3",
                architecture
            ));
        }
    };

    Ok((model, tokenizer, device.clone(), dtype))
}

// Made with Bob

/// Find a GGUF file for the given model ID
/// Checks both HuggingFace cache and local paths
/// Infer base model repository from GGUF filename or model_id
/// GGUF files don't include tokenizers, so we need to get them from the base model
fn infer_base_model_from_gguf(gguf_path: &std::path::Path, model_id: &str) -> String {
    // Get filename from path
    let filename = gguf_path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    // Check if model_id ends with -GGUF (common pattern for GGUF repos)
    // e.g., "Qwen/Qwen2.5-1.5B-Instruct-GGUF" -> "Qwen/Qwen2.5-1.5B-Instruct"
    // or "unsloth/Qwen3-1.7B-GGUF" -> "Qwen/Qwen3-1.7B"
    if model_id.ends_with("-GGUF") {
        let base = model_id.trim_end_matches("-GGUF");
        // If it's from a third-party repo like unsloth, map to official Qwen repo
        if base.contains('/') && !base.starts_with("Qwen/") {
            // Extract just the model name part (e.g., "Qwen3-1.7B" from "unsloth/Qwen3-1.7B")
            let model_name = base.split('/').next_back().unwrap_or(base);

            // Special case: Qwen3 MoE models (A3B variants) map to specific base models
            if model_name.contains("Qwen3")
                && (model_name.contains("A3B") || model_name.contains("a3b"))
            {
                // All Qwen3 A3B variants use the same base model for tokenizer
                return "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string();
            }

            return format!("Qwen/{}", model_name);
        }
        return base.to_string();
    }

    // Try to extract model info from filename or model_id
    // Common patterns: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    let combined = format!("{} {}", filename, model_id);

    if combined.contains("Qwen2.5-0.5B") {
        return "Qwen/Qwen2.5-0.5B-Instruct".to_string();
    } else if combined.contains("Qwen2.5-1.5B") {
        return "Qwen/Qwen2.5-1.5B-Instruct".to_string();
    } else if combined.contains("Qwen2.5-3B") {
        return "Qwen/Qwen2.5-3B-Instruct".to_string();
    } else if combined.contains("Qwen2.5-7B") {
        return "Qwen/Qwen2.5-7B-Instruct".to_string();
    } else if combined.contains("Qwen3") {
        // Check for MoE models first (e.g., 30B-A3B)
        if combined.contains("30B-A3B") || combined.contains("30b-a3b") {
            return "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string();
        } else if combined.contains("A3B") || combined.contains("a3b") {
            // Default A3B variant
            return "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string();
        } else if combined.contains("0.6B") {
            return "Qwen/Qwen3-0.6B".to_string();
        } else if combined.contains("1.7B") {
            return "Qwen/Qwen3-1.7B".to_string();
        } else if combined.contains("4B") {
            return "Qwen/Qwen3-4B".to_string();
        } else if combined.contains("8B") {
            return "Qwen/Qwen3-8B".to_string();
        } else if combined.contains("14B") {
            return "Qwen/Qwen3-14B".to_string();
        } else if combined.contains("32B") {
            return "Qwen/Qwen3-32B".to_string();
        }
    }

    // Default fallback - try the model_id as-is
    model_id.to_string()
}

fn find_gguf_file(model_id: &str) -> anyhow::Result<std::path::PathBuf> {
    use std::path::PathBuf;

    // Check if model_id is a direct path to a GGUF file
    let direct_path = PathBuf::from(model_id);
    if direct_path.exists() && direct_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        return Ok(direct_path);
    }

    // Check in HuggingFace cache
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| anyhow::anyhow!("Could not find home directory"))?;
    let cache_dir = std::path::PathBuf::from(home_dir).join(".cache/huggingface/hub");

    // Convert model_id to cache directory format (e.g., "Qwen/Qwen2.5-0.5B" -> "models--Qwen--Qwen2.5-0.5B")
    let cache_model_dir = format!("models--{}", model_id.replace('/', "--"));
    let model_cache_path = cache_dir.join(&cache_model_dir);

    if model_cache_path.exists() {
        // Look for GGUF files in snapshots
        for entry in std::fs::read_dir(model_cache_path.join("snapshots"))? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                for file in std::fs::read_dir(entry.path())? {
                    let file = file?;
                    if file.path().extension().and_then(|s| s.to_str()) == Some("gguf") {
                        return Ok(file.path());
                    }
                }
            }
        }
    }

    Err(anyhow::anyhow!(
        "No GGUF file found for model: {}",
        model_id
    ))
}

/// Load a GGUF quantized model
fn load_gguf_model(
    gguf_path: &std::path::Path,
    model_id: &str,
    device: Device,
) -> anyhow::Result<(Box<dyn CandleModel>, Tokenizer, Device, DType)> {
    // Load tokenizer from HuggingFace (GGUF files don't include tokenizer)
    // Determine base model from GGUF filename or model_id
    let base_model_id = infer_base_model_from_gguf(gguf_path, model_id);

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(base_model_id.to_string());
    let tokenizer_path = repo.get("tokenizer.json")?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Detect architecture from model_id or GGUF metadata
    // For now, we'll use a simple heuristic based on model name
    let model: Box<dyn CandleModel> = if model_id.to_lowercase().contains("qwen3") {
        // Check if this is a MoE model (e.g., A3B)
        if model_id.to_lowercase().contains("a3b") || model_id.to_lowercase().contains("moe") {
            Box::new(QuantizedQwen3MoeModelWrapper::load(
                gguf_path,
                device.clone(),
            )?)
        } else {
            Box::new(QuantizedQwen3ModelWrapper::load(gguf_path, device.clone())?)
        }
    } else if model_id.to_lowercase().contains("qwen") {
        Box::new(QuantizedQwen2ModelWrapper::load(gguf_path, device.clone())?)
    } else {
        return Err(anyhow::anyhow!(
            "Quantized model loading not yet implemented for architecture: {}. Currently supported: Qwen2/Qwen2.5/Qwen3/Qwen3-MoE",
            model_id
        ));
    };

    // GGUF models don't have a specific dtype (they're quantized)
    // Return F16 as a placeholder since it's not used for quantized models
    Ok((model, tokenizer, device, DType::F16))
}
