use crate::{
    SpnlResult,
    ir::{Map, Repeat},
};

#[cfg(any(feature = "ollama", feature = "openai", feature = "gemini"))]
use crate::generate::backend::openai::Provider::*;

pub mod backend;

#[derive(thiserror::Error, Debug)]
#[error("Model not found")]
pub struct ModelNotFoundError;

pub async fn map(spec: &Map, mmp: Option<&indicatif::MultiProgress>, prepare: bool) -> SpnlResult {
    let mymp = indicatif::MultiProgress::new();
    let mp = mmp.or({
        if spec.inputs.len() > 1 {
            Some(&mymp)
        } else {
            None
        }
    });
    match spec.metadata.model.splitn(2, '/').collect::<Vec<_>>()[..] {
        #[cfg(feature = "ollama")]
        ["ollama", m] => {
            backend::openai::generate_completion(Ollama, spec.with_model(m)?, mp, prepare).await
        }

        #[cfg(feature = "openai")]
        ["openai", m] => {
            backend::openai::generate_completion(OpenAI, spec.with_model(m)?, mp, prepare).await
        }

        #[cfg(feature = "gemini")]
        ["gemini", m] => {
            backend::openai::generate_completion(Gemini, spec.with_model(m)?, mp, prepare).await
        }

        #[cfg(feature = "spnl-api")]
        ["spnl", m] => {
            backend::openai::generate_completion(OpenAI, spec.with_model(m)?, mp, prepare).await
            // TODO "native" spnl support for Map backend::spnl::generate_completion(spec.with_model(m)?, mp, prepare).await,
        }

        _ => Err(ModelNotFoundError.into()),
    }
}

pub async fn generate(
    spec: Repeat,
    mp: Option<&indicatif::MultiProgress>,
    prepare: bool,
) -> SpnlResult {
    match spec
        .generate
        .metadata
        .model
        .splitn(2, '/')
        .collect::<Vec<_>>()[..]
    {
        #[cfg(feature = "ollama")]
        ["ollama", m] => {
            backend::openai::generate_chat(Ollama, spec.with_model(m)?, mp, prepare).await
        }

        #[cfg(feature = "openai")]
        ["openai", m] => {
            backend::openai::generate_chat(OpenAI, spec.with_model(m)?, mp, prepare).await
        }

        #[cfg(feature = "gemini")]
        ["gemini", m] => {
            backend::openai::generate_chat(Gemini, spec.with_model(m)?, mp, prepare).await
        }

        #[cfg(feature = "spnl-api")]
        ["spnl", m] => backend::spnl::generate(spec.with_model(m)?, mp, prepare).await,

        _ => Err(ModelNotFoundError.into()),
    }
}
