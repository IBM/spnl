use super::Generate;

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Bulk {
    Repeat(Repeat),
}

/// Bulk operation: generate `n` outputs for the given `generate` specification
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Repeat {
    /// The number of outputs to generate
    pub n: usize,

    /// The specification of what to generate
    pub generate: Generate,
}
