use super::{Generate, GenerateMetadata};

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Bulk {
    Repeat(Repeat),

    Map(Map),
}

/// Bulk operation: generate `n` outputs for the given `generate` specification
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Repeat {
    /// The number of outputs to generate
    pub n: usize,

    /// The specification of what to generate
    pub generate: Generate,
}

/// Bulk operation: map the generate operation across the given inputs, using the given metadata
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Map {
    /// The metadata governing the content generation
    pub metadata: GenerateMetadata,

    /// Generate one output for each input in this list
    pub inputs: Vec<String>,
}
