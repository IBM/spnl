use super::Generate;

/// Bulk operation: generate `n` outputs for the given `generate` specification
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Repeat {
    pub n: usize,
    pub generate: Generate,
}
