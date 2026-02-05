use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::gemma3::{Config as Gemma3Config, Model as Gemma3Model};
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::model::{CandleModel, ModelForward};

#[derive(Debug, Deserialize)]
pub struct Gemma3GenericConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub rms_norm_eps: Option<f64>,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    pub hidden_act: Option<String>,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: Option<usize>,
}

impl From<Gemma3GenericConfig> for Gemma3Config {
    fn from(config: Gemma3GenericConfig) -> Self {
        let hidden_activation = match config.hidden_act.as_deref() {
            Some("gelu_pytorch_tanh") => candle_nn::Activation::GeluPytorchTanh,
            Some("gelu") => candle_nn::Activation::Gelu,
            Some("relu") => candle_nn::Activation::Relu,
            _ => candle_nn::Activation::GeluPytorchTanh, // Gemma3 default
        };

        Gemma3Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            head_dim: config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads),
            rms_norm_eps: config.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: config.rope_theta.unwrap_or(10000.0),
            max_position_embeddings: config.max_position_embeddings.unwrap_or(8192),
            attention_bias: config.attention_bias.unwrap_or(false),
            hidden_activation,
            attn_logit_softcapping: config.attn_logit_softcapping,
            final_logit_softcapping: config.final_logit_softcapping,
            query_pre_attn_scalar: config.query_pre_attn_scalar.unwrap_or(256),
            rope_local_base_freq: 10000.0, // Default value
            sliding_window: 4096,          // Default value
            sliding_window_pattern: 1024,  // Default value
        }
    }
}

pub struct Gemma3ModelWrapper {
    model: Gemma3Model,
    config: Gemma3Config,
    cache_position: usize, // Track current cache length for reuse
}

impl Gemma3ModelWrapper {
    pub fn load(vb: VarBuilder, config: Gemma3Config) -> anyhow::Result<Self> {
        let use_flash_attn = false; // Disable flash attention for compatibility
        let model = Gemma3Model::new(use_flash_attn, &config, vb)?;
        Ok(Self {
            model,
            config,
            cache_position: 0,
        })
    }
}

impl ModelForward for Gemma3ModelWrapper {
    fn forward_pass(&mut self, input: &Tensor, position: usize) -> anyhow::Result<Tensor> {
        let result = self.model.forward(input, position)?;
        // Update cache position to track what's cached
        let input_len = input.dim(1)?;
        self.cache_position = position + input_len;
        Ok(result)
    }

    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
        self.cache_position = 0;
    }

    fn get_cache_length(&self) -> usize {
        self.cache_position
    }

    fn clear_cache_after(&mut self, position: usize) {
        // For now, we don't have a way to partially clear Candle's cache
        // So we clear everything if position is less than current cache
        if position < self.cache_position {
            self.clear_cache();
        }
    }
}

impl CandleModel for Gemma3ModelWrapper {
    fn generate(
        &mut self,
        tokens: &[u32],
        config: super::model::GenerateConfig,
        token_callback: Option<&mut super::model::TokenCallback>,
    ) -> anyhow::Result<String> {
        let eos_token = self.eos_token_id(config.tokenizer);
        super::model::generate_text(self, tokens, eos_token, config, token_callback)
    }

    fn eos_token_id(&self, tokenizer: &Tokenizer) -> u32 {
        // Try to get EOS token from tokenizer, fallback to common Gemma3 EOS token
        tokenizer
            .token_to_id("<eos>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<end_of_turn>"))
            .unwrap_or(1) // Gemma3 typically uses 1 as EOS
    }
}

// Made with Bob
