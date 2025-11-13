use crate::ir::{Generate, Message, Query};

//#[pyclass]
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

//#[pyclass]
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct SingleGenerate {
    pub model: String,
    pub input: NonGenerateInput,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
}

//#[pyclass]
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct SingleGenerateQuery {
    pub g: SingleGenerate,
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

impl From<SingleGenerateQuery> for Query {
    fn from(q: SingleGenerateQuery) -> Self {
        Self::Generate(Generate {
            model: q.g.model.clone(),
            input: Box::new(q.g.input.clone().into()),
            max_tokens: q.g.max_tokens,
            temperature: q.g.temperature,
        })
    }
}
