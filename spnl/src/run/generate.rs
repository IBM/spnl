use ::std::fmt;
use indicatif::MultiProgress;

use crate::{Query, run::result::SpnlResult};

pub struct ModelNotFoundError;

// Implement StdError for ModelNotFoundError
impl ::std::error::Error for ModelNotFoundError {
    fn description(&self) -> &str {
        "Model not found"
    }
}

// Implement std::fmt::Display for ModelNotFoundError
impl fmt::Display for ModelNotFoundError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Model not found") // user-facing output
    }
}

// Implement std::fmt::Debug for ModelNotFoundError
impl fmt::Debug for ModelNotFoundError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Model not found {{ file: {}, line: {} }}",
            file!(),
            line!()
        ) // programmer-facing output
    }
}

pub async fn generate(
    model: &str,
    input: &Query,
    max_tokens: &Option<i32>,
    temp: &Option<f32>,
    mp: Option<&MultiProgress>,
    prepare: bool,
) -> SpnlResult {
    match model {
        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama/") => {
            crate::run::backend::openai::generate(
                crate::run::backend::openai::Provider::Ollama,
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
            crate::run::backend::openai::generate(
                crate::run::backend::openai::Provider::Ollama,
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
            crate::run::backend::openai::generate(
                crate::run::backend::openai::Provider::OpenAI,
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
            crate::run::backend::openai::generate(
                crate::run::backend::openai::Provider::Gemini,
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
            crate::run::backend::spnl::generate(&m[5..], input, max_tokens, temp, mp, prepare).await
        }

        _ => Err(Box::from(ModelNotFoundError)),
    }
}
