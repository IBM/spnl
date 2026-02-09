use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM as Qwen3Model};
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::model::{CandleModel, ModelForward};

#[derive(Debug, Deserialize)]
pub struct Qwen3GenericConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
    pub rope_theta: Option<f64>,
    pub rms_norm_eps: Option<f64>,
    pub use_sliding_window: Option<bool>,
    pub hidden_act: Option<String>,
    pub attention_bias: Option<bool>,
    pub head_dim: Option<usize>,
}

impl From<Qwen3GenericConfig> for Qwen3Config {
    fn from(config: Qwen3GenericConfig) -> Self {
        let hidden_act = match config.hidden_act.as_deref() {
            Some("silu") => candle_nn::Activation::Silu,
            Some("gelu") => candle_nn::Activation::Gelu,
            Some("relu") => candle_nn::Activation::Relu,
            _ => candle_nn::Activation::Silu, // default
        };

        let num_key_value_heads = config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads);
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        Qwen3Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings: config.max_position_embeddings.unwrap_or(32768),
            sliding_window: config.sliding_window,
            max_window_layers: config.max_window_layers.unwrap_or(config.num_hidden_layers),
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
            rope_theta: config.rope_theta.unwrap_or(1000000.0),
            rms_norm_eps: config.rms_norm_eps.unwrap_or(1e-6),
            use_sliding_window: config.use_sliding_window.unwrap_or(false),
            attention_bias: config.attention_bias.unwrap_or(false),
            hidden_act,
        }
    }
}

pub struct Qwen3ModelWrapper {
    model: Qwen3Model,
    config: Qwen3Config,
    cache_position: usize, // Track current cache length for reuse
}

impl Qwen3ModelWrapper {
    pub fn load(vb: VarBuilder, config: Qwen3Config) -> anyhow::Result<Self> {
        // Qwen3 ModelForCausalLM adds "model." prefix internally
        let model = Qwen3Model::new(&config, vb)?;
        Ok(Self {
            model,
            config,
            cache_position: 0,
        })
    }
}

impl ModelForward for Qwen3ModelWrapper {
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

impl CandleModel for Qwen3ModelWrapper {
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
        // Qwen3 models typically use <|im_end|> for chat format
        tokenizer
            .token_to_id("<|im_end|>") // Qwen chat format (priority)
            .or_else(|| tokenizer.token_to_id("<|endoftext|>")) // Standard EOS
            .or_else(|| tokenizer.token_to_id("</s>")) // Llama-style EOS
            .or_else(|| tokenizer.token_to_id("<|end|>")) // Alternative
            .unwrap_or(151643) // Default Qwen EOS token ID
    }
}

// Made with Bob
