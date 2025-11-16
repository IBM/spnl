use crate::{
    SpnlResult,
    ir::Query,
    ir::{Generate, GenerateMetadata},
};

#[cfg(any(feature = "ollama", feature = "openai", feature = "gemini"))]
use crate::generate::backend::openai::Provider::*;

pub mod backend;

#[derive(thiserror::Error, Debug)]
#[error("Model not found")]
pub struct ModelNotFoundError;

pub fn is_span_enabled(model: &str) -> bool {
    // for now...
    model.starts_with("spnl/")
}

pub async fn map(
    _metadata: &GenerateMetadata,
    _inputs: &[String],
    _mp: Option<&indicatif::MultiProgress>,
) -> SpnlResult {
    Ok(Query::Message(crate::ir::Message::User("".into())))
}

pub async fn repeat(
    _n: usize,
    _generate: &Generate,
    _mp: Option<&indicatif::MultiProgress>,
) -> SpnlResult {
    Ok(Query::Message(crate::ir::Message::User("".into())))
}

pub async fn generate(
    spec: Generate,
    mp: Option<&indicatif::MultiProgress>,
    prepare: bool,
) -> SpnlResult {
    match spec.metadata.model.splitn(2, '/').collect::<Vec<_>>()[..] {
        #[cfg(feature = "ollama")]
        ["ollama/", m] => backend::openai::generate(Ollama, spec.with_model(m)?, mp, prepare).await,

        #[cfg(feature = "openai")]
        ["openai/", m] => backend::openai::generate(OpenAI, spec.with_model(m)?, mp, prepare).await,

        #[cfg(feature = "gemini")]
        ["gemini/", m] => backend::openai::generate(Gemini, spec.with_model(m)?, mp, prepare).await,

        #[cfg(feature = "spnl-api")]
        ["spnl", m] => backend::spnl::generate(spec.with_model(m)?, mp, prepare).await,

        _ => Err(ModelNotFoundError.into()),
    }
}
