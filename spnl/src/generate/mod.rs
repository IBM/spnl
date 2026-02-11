use crate::{
    SpnlResult,
    ir::{Map, Repeat},
};

#[cfg(any(feature = "ollama", feature = "openai", feature = "gemini"))]
use crate::generate::backend::openai::Provider::*;

pub mod backend;

mod options;
pub use options::*;

#[derive(thiserror::Error, Debug)]
#[error("Model not found")]
pub struct ModelNotFoundError;

pub async fn map(
    spec: &Map,
    mmp: Option<&indicatif::MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
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
            backend::openai::generate_completion(Ollama, spec.with_model(m)?, mp, options).await
        }

        #[cfg(feature = "openai")]
        ["openai", m] => {
            backend::openai::generate_completion(OpenAI, spec.with_model(m)?, mp, options).await
        }

        #[cfg(feature = "gemini")]
        ["gemini", m] => {
            backend::openai::generate_completion(Gemini, spec.with_model(m)?, mp, options).await
        }

        #[cfg(feature = "spnl-api")]
        ["spnl", m] => {
            backend::spnl::generate(backend::spnl::Spec::Map(spec.with_model(m)?), mp, options)
                .await
            // FYI this is what we would do to invoke via the openai bulk api directly: backend::openai::generate_completion(OpenAI, spec.with_model(m)?, mp, options).await
        }

        #[cfg(feature = "local")]
        ["local", m] => {
            backend::mistralrs::generate_completion(spec.with_model(m)?, mp, options).await
        }

        _ => Err(ModelNotFoundError.into()),
    }
}

pub async fn generate(
    spec: Repeat,
    mp: Option<&indicatif::MultiProgress>,
    options: &GenerateOptions,
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
            backend::openai::generate_chat(Ollama, spec.with_model(m)?, mp, options).await
        }

        #[cfg(feature = "openai")]
        ["openai", m] => {
            backend::openai::generate_chat(OpenAI, spec.with_model(m)?, mp, options).await
        }

        #[cfg(feature = "gemini")]
        ["gemini", m] => {
            backend::openai::generate_chat(Gemini, spec.with_model(m)?, mp, options).await
        }

        #[cfg(feature = "spnl-api")]
        ["spnl", m] => {
            backend::spnl::generate(
                backend::spnl::Spec::Repeat(spec.with_model(m)?),
                mp,
                options,
            )
            .await
        }

        #[cfg(feature = "local")]
        ["local", m] => backend::mistralrs::generate_chat(spec.with_model(m)?, mp, options).await,

        _ => Err(ModelNotFoundError.into()),
    }
}
