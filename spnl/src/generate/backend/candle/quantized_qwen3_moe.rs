use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3_moe as quantized;
use tokenizers::Tokenizer;

use super::model::{CandleModel, GenerateConfig, ModelForward, TokenCallback, generate_text};

/// Wrapper for quantized Qwen3 MoE models (Q4_K_M, Q8_0, etc.)
pub struct QuantizedQwen3MoeModelWrapper {
    model: quantized::GGUFQWenMoE,
    #[allow(dead_code)]
    device: Device,
}

impl QuantizedQwen3MoeModelWrapper {
    /// Load a quantized Qwen3 MoE model from a GGUF file
    ///
    /// Note: Quantized MoE models require CUDA backend. They are not supported on Metal or CPU.
    pub fn load(gguf_path: &std::path::Path, device: Device) -> anyhow::Result<Self> {
        // Check if device is CUDA
        if !device.is_cuda() {
            return Err(anyhow::anyhow!(
                "Quantized Qwen3 MoE models require CUDA backend. \
                The moe_gemm_gguf operation is only implemented for CUDA. \
                Current device: {:?}. \
                Please use a CUDA-enabled GPU or use the non-quantized Qwen3 MoE model instead.",
                device
            ));
        }

        use candle_core::quantized::gguf_file;

        // Open and read GGUF file
        let mut file = std::fs::File::open(gguf_path)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Log quantization info
        /* eprintln!("Loading quantized Qwen3 MoE model from GGUF:");
        if let Some(tensor_info) = content.tensor_infos.values().next() {
            eprintln!("  Quantization format: {:?}", tensor_info.ggml_dtype);
        } */

        // Load quantized model weights
        // Use F16 as dtype for quantized models
        let model = quantized::GGUFQWenMoE::from_gguf(content, &mut file, &device, DType::F16)?;

        Ok(Self { model, device })
    }
}

/// Forward pass trait implementation for quantized Qwen3 MoE
impl super::model::ModelForward for QuantizedQwen3MoeModelWrapper {
    fn forward_pass(&mut self, input: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input, pos)?)
    }

    fn clear_cache(&mut self) {
        // Quantized MoE models don't expose clear_kv_cache
        // Cache is managed internally
    }

    fn get_cache_length(&self) -> usize {
        // Quantized models track cache length internally
        // For now, return 0 to indicate we should check the model
        0
    }

    fn max_position_embeddings(&self) -> usize {
        // Qwen3 MoE default max position embeddings
        // This should ideally come from the model config
        32768
    }
}

impl CandleModel for QuantizedQwen3MoeModelWrapper {
    fn generate(
        &mut self,
        tokens: &[u32],
        config: GenerateConfig,
        token_callback: Option<&mut TokenCallback>,
    ) -> anyhow::Result<String> {
        // Always clear cache for quantized models since we can't track cache length
        self.clear_cache();
        let eos_token = self.eos_token_id(config.tokenizer);
        generate_text(self, tokens, eos_token, config, token_callback)
    }

    fn eos_token_id(&self, tokenizer: &Tokenizer) -> u32 {
        // Qwen3 MoE models typically use <|endoftext|> or <|im_end|> as EOS
        tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(151643) // Qwen3 default EOS token (same as Qwen2)
    }
}

// Made with Bob
