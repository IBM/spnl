use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2 as quantized;
use tokenizers::Tokenizer;

use super::model::{CandleModel, GenerateConfig, TokenCallback, generate_text};

/// Wrapper for quantized Qwen2 models (Q4_K_M, Q8_0, etc.)
pub struct QuantizedQwen2ModelWrapper {
    model: quantized::ModelWeights,
    #[allow(dead_code)]
    device: Device,
}

impl QuantizedQwen2ModelWrapper {
    /// Load a quantized Qwen2 model from a GGUF file
    pub fn load(gguf_path: &std::path::Path, device: Device) -> anyhow::Result<Self> {
        use candle_core::quantized::gguf_file;

        // Open and read GGUF file
        let mut file = std::fs::File::open(gguf_path)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Log quantization info
        /* eprintln!("Loading quantized Qwen2 model from GGUF:");
        if let Some(tensor_info) = content.tensor_infos.values().next() {
            eprintln!("  Quantization format: {:?}", tensor_info.ggml_dtype);
        } */

        // Load quantized model weights
        let model = quantized::ModelWeights::from_gguf(content, &mut file, &device)?;

        Ok(Self { model, device })
    }
}

/// Forward pass trait implementation for quantized Qwen2
impl super::model::ModelForward for QuantizedQwen2ModelWrapper {
    fn forward_pass(&mut self, input: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input, pos)?)
    }

    fn clear_cache(&mut self) {
        // Quantized models don't expose clear_kv_cache
        // Cache is managed internally
    }

    fn get_cache_length(&self) -> usize {
        // Quantized models track cache length internally
        // For now, return 0 to indicate we should check the model
        0
    }

    fn max_position_embeddings(&self) -> usize {
        // Qwen2.5 default max position embeddings
        // This should ideally come from the model config
        32768
    }
}

impl CandleModel for QuantizedQwen2ModelWrapper {
    fn generate(
        &mut self,
        tokens: &[u32],
        config: GenerateConfig,
        token_callback: Option<&mut TokenCallback>,
    ) -> anyhow::Result<String> {
        let eos_token = self.eos_token_id(config.tokenizer);
        generate_text(self, tokens, eos_token, config, token_callback)
    }

    fn eos_token_id(&self, tokenizer: &Tokenizer) -> u32 {
        // Try to get EOS token from tokenizer
        tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .unwrap_or(151643) // Qwen2 default EOS token
    }
}

// Made with Bob
