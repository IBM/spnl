use crate::ir::{GenerateMetadata, Map, Message};

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Bulk {
    Repeat(Repeat),

    Map(Map),
}

/// Bulk operation: generate `n` outputs for the given `generate` specification
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct Repeat {
    /// The number of outputs to generate
    pub n: u8,

    /// The specification of what to generate
    pub generate: SingleGenerate,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum NonGenerateInput {
    /// Execute serially
    Seq(Vec<NonGenerateInput>),

    /// Execute in parallel
    Par(Vec<NonGenerateInput>),

    /// Commutative
    Plus(Vec<NonGenerateInput>),

    /// Non-Commutative
    Cross(Vec<NonGenerateInput>),

    /// Some sort of message
    #[serde(untagged)]
    Message(Message),
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct SingleGenerate {
    #[serde(flatten)]
    pub metadata: GenerateMetadata,

    pub input: NonGenerateInput,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum SingleGenerateQuery {
    #[serde(rename = "g")]
    SingleGenerate(SingleGenerate),

    /// Some kind of bulk operation
    #[serde(untagged)]
    Bulk(Bulk),
}
