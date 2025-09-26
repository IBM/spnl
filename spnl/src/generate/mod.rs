use crate::{Query, SpnlResult};

pub mod backend;

#[derive(thiserror::Error, Debug)]
#[error("Model not found")]
pub struct ModelNotFoundError;

pub fn is_span_enabled(model: &str) -> bool {
    // for now...
    model.starts_with("spnl/")
}

pub async fn generate(
    model: &str,
    input: &Query,
    max_tokens: &Option<i32>,
    temp: &Option<f32>,
    mp: Option<&indicatif::MultiProgress>,
    prepare: bool,
) -> SpnlResult {
    match model {
        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama/") => {
            crate::generate::backend::openai::generate(
                crate::generate::backend::openai::Provider::Ollama,
                &m[7..],
                input,
                max_tokens,
                temp,
                mp,
                prepare,
            )
            .await
        }

        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama_chat/") => {
            crate::generate::backend::openai::generate(
                crate::generate::backend::openai::Provider::Ollama,
                &m[12..],
                input,
                max_tokens,
                temp,
                mp,
                prepare,
            )
            .await
        }

        #[cfg(feature = "openai")]
        m if m.starts_with("openai/") => {
            crate::generate::backend::openai::generate(
                crate::generate::backend::openai::Provider::OpenAI,
                &m[7..],
                input,
                max_tokens,
                temp,
                mp,
                prepare,
            )
            .await
        }

        #[cfg(feature = "gemini")]
        m if m.starts_with("gemini/") => {
            crate::generate::backend::openai::generate(
                crate::generate::backend::openai::Provider::Gemini,
                &m[7..],
                input,
                max_tokens,
                temp,
                mp,
                prepare,
            )
            .await
        }

        #[cfg(feature = "spnl-api")]
        m if m.starts_with("spnl/") => {
            crate::generate::backend::spnl::generate(&m[5..], input, max_tokens, temp, mp, prepare)
                .await
        }

        _ => Err(ModelNotFoundError.into()),
    }
}
