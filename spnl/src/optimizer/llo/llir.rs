use crate::ir::{GenerateMetadata, Message, Query};

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum NonGenerateInput {
    /// Reduce
    Plus(Vec<NonGenerateInput>),

    /// Map
    Cross(Vec<NonGenerateInput>),

    /// Execute serially
    Seq(Vec<NonGenerateInput>),

    /// Execute in parallel
    Par(Vec<NonGenerateInput>),

    /// Some sort of message
    #[serde(untagged)]
    Message(Message),
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct SingleGenerate {
    pub input: NonGenerateInput,
    pub metadata: GenerateMetadata,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) enum SingleGenerateQuery {
    #[serde(rename = "g")]
    SingleGenerate(SingleGenerate),
}

impl From<NonGenerateInput> for Query {
    fn from(input: NonGenerateInput) -> Self {
        match input {
            NonGenerateInput::Message(m) => Query::Message(m),
            NonGenerateInput::Plus(v) => Query::Plus(v.into_iter().map(|m| m.into()).collect()),
            NonGenerateInput::Cross(v) => Query::Cross(v.into_iter().map(|m| m.into()).collect()),
            NonGenerateInput::Seq(v) => Query::Seq(v.into_iter().map(|m| m.into()).collect()),
            NonGenerateInput::Par(v) => Query::Par(v.into_iter().map(|m| m.into()).collect()),
        }
    }
}
