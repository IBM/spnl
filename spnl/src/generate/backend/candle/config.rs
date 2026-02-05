use serde::Deserialize;

// Generic config structure that can be deserialized from any model's config.json
#[derive(Debug, Deserialize)]
pub(crate) struct GenericConfig {
    pub architectures: Option<Vec<String>>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: Option<f64>,
    pub rope_theta: Option<f64>,
    pub max_position_embeddings: Option<usize>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    // Qwen2/Qwen3-specific fields
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
    pub use_sliding_window: Option<bool>,
    pub hidden_act: Option<String>,
    pub attention_bias: Option<bool>,
    pub head_dim: Option<usize>,
    // MoE-specific fields
    pub num_experts_per_tok: Option<usize>,
    pub num_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub decoder_sparse_step: Option<usize>,
    pub norm_topk_prob: Option<bool>,
}

/// Detect model architecture from config
pub(crate) fn detect_architecture(config: &GenericConfig) -> anyhow::Result<String> {
    if let Some(ref architectures) = config.architectures
        && let Some(arch) = architectures.first()
    {
        return Ok(arch.clone());
    }
    Err(anyhow::anyhow!("No architecture specified in config.json"))
}

// Made with Bob
