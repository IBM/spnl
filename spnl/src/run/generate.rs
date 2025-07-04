use indicatif::MultiProgress;

use crate::{Query, run::result::SpnlResult};

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
            crate::run::backend::ollama::generate(&m[7..], input, max_tokens, temp, mp, prepare)
                .await
        }

        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama_chat/") => {
            crate::run::backend::ollama::generate(&m[12..], input, max_tokens, temp, mp, prepare)
                .await
        }

        #[cfg(feature = "openai")]
        m if m.starts_with("openai/") => {
            crate::run::backend::openai::generate(&m[7..], input, max_tokens, temp, mp, prepare)
                .await
        }

        #[cfg(feature = "spnl-api")]
        m if m.starts_with("spnl/") => {
            crate::run::backend::spnl::generate(&m[5..], input, max_tokens, temp, mp, prepare).await
        }

        _ => todo!("Unknown model {model}"),
    }
}
