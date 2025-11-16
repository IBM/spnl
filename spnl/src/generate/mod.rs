use crate::{
    SpnlResult,
    ir::Query,
    ir::{Generate, GenerateBuilder, GenerateMetadata, GenerateMetadataBuilder},
};

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
    metadata: &GenerateMetadata,
    input: &Query,
    mp: Option<&indicatif::MultiProgress>,
    prepare: bool,
) -> SpnlResult {
    match &metadata.model {
        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama/") => {
            crate::generate::backend::openai::generate(
                crate::generate::backend::openai::Provider::Ollama,
                GenerateBuilder::default()
                    .input(Box::new(input.clone()))
                    .metadata(
                        GenerateMetadataBuilder::from(metadata)
                            .model(&m[7..])
                            .build()?,
                    )
                    .build()?,
                mp,
                prepare,
            )
            .await
        }

        #[cfg(feature = "openai")]
        m if m.starts_with("openai/") => {
            crate::generate::backend::openai::generate(
                crate::generate::backend::openai::Provider::OpenAI,
                GenerateBuilder::default()
                    .input(Box::new(input.clone()))
                    .metadata(
                        GenerateMetadataBuilder::from(metadata)
                            .model(&m[7..])
                            .build()?,
                    )
                    .build()?,
                mp,
                prepare,
            )
            .await
        }

        #[cfg(feature = "gemini")]
        m if m.starts_with("gemini/") => {
            crate::generate::backend::openai::generate(
                crate::generate::backend::openai::Provider::Gemini,
                GenerateBuilder::default()
                    .input(Box::new(input.clone()))
                    .metadata(
                        GenerateMetadataBuilder::from(metadata)
                            .model(&m[7..])
                            .build()?,
                    )
                    .build()?,
                mp,
                prepare,
            )
            .await
        }

        #[cfg(feature = "spnl-api")]
        m if m.starts_with("spnl/") => {
            crate::generate::backend::spnl::generate(
                GenerateBuilder::default()
                    .input(Box::new(input.clone()))
                    .metadata(
                        GenerateMetadataBuilder::from(metadata)
                            .model(&m[5..])
                            .build()?,
                    )
                    .build()?,
                mp,
                prepare,
            )
            .await
        }

        _ => Err(ModelNotFoundError.into()),
    }
}
