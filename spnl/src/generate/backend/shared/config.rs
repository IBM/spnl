use serde::{Deserialize, Deserializer};

/// Custom deserializer for eos_token_id that handles both single value and array
fn deserialize_eos_token_id<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum EosTokenId {
        Single(u32),
        Multiple(Vec<u32>),
    }

    let value = Option::<EosTokenId>::deserialize(deserializer)?;
    Ok(value.and_then(|v| match v {
        EosTokenId::Single(id) => Some(id),
        EosTokenId::Multiple(ids) => ids.first().copied(), // Take first if array
    }))
}

// Generic config structure that can be deserialized from any model's config.json
#[derive(Debug, Deserialize)]
pub struct GenericConfig {
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
    #[serde(deserialize_with = "deserialize_eos_token_id")]
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
pub fn detect_architecture(config: &GenericConfig) -> anyhow::Result<String> {
    if let Some(ref architectures) = config.architectures
        && let Some(arch) = architectures.first()
    {
        return Ok(arch.clone());
    }
    Err(anyhow::anyhow!("No architecture specified in config.json"))
}

// Made with Bob
