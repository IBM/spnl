use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3 as quantized;
use tokenizers::Tokenizer;

use super::model::{CandleModel, GenerateConfig, TokenCallback, generate_text};

/// Wrapper for quantized Qwen3 models (Q4_K_M, Q8_0, etc.)
pub struct QuantizedQwen3ModelWrapper {
    model: quantized::ModelWeights,
    #[allow(dead_code)]
    device: Device,
    cache_position: usize, // Track current cache length for reuse
}

impl QuantizedQwen3ModelWrapper {
    /// Load a quantized Qwen3 model from a GGUF file
    pub fn load(gguf_path: &std::path::Path, device: Device) -> anyhow::Result<Self> {
        use candle_core::quantized::gguf_file;

        // Open and read GGUF file
        let mut file = std::fs::File::open(gguf_path)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Log quantization info
        /* eprintln!("Loading quantized Qwen3 model from GGUF:");
        if let Some(tensor_info) = content.tensor_infos.values().next() {
            eprintln!("  Quantization format: {:?}", tensor_info.ggml_dtype);
        } */

        // Load quantized model weights
        let model = quantized::ModelWeights::from_gguf(content, &mut file, &device)?;

        Ok(Self {
            model,
            device,
            cache_position: 0,
        })
    }
}

/// Forward pass trait implementation for quantized Qwen3
impl super::model::ModelForward for QuantizedQwen3ModelWrapper {
    fn forward_pass(&mut self, input: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        let result = self.model.forward(input, pos)?;
        // Update cache position to track what's cached
        let input_len = input.dim(1)?;
        self.cache_position = pos + input_len;
        Ok(result)
    }

    fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
        self.cache_position = 0;
    }

    fn get_cache_length(&self) -> usize {
        self.cache_position
    }

    fn max_position_embeddings(&self) -> usize {
        // Qwen3 default max position embeddings
        // This should ideally come from the model config
        32768
    }
}

impl CandleModel for QuantizedQwen3ModelWrapper {
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
        // Try to get EOS token from tokenizer - try multiple common EOS tokens
        tokenizer
            .token_to_id("<|im_end|>") // Qwen chat format
            .or_else(|| tokenizer.token_to_id("<|endoftext|>")) // Standard EOS
            .or_else(|| tokenizer.token_to_id("</s>")) // Llama-style EOS
            .or_else(|| tokenizer.token_to_id("<|end|>")) // Alternative
            .unwrap_or(151643) // Qwen3 default EOS token (same as Qwen2)
    }
}

// Made with Bob
