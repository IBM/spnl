use crate::Query;
use crate::run::result::SpnlError;

pub enum EmbedData {
    Query(Query),
    Vec(Vec<String>),
}

pub async fn embed(embedding_model: &String, data: EmbedData) -> Result<Vec<Vec<f32>>, SpnlError> {
    match embedding_model {
        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama/") => {
            crate::run::backend::openai::embed(
                crate::run::backend::openai::Provider::Ollama,
                &m[7..],
                &data,
            )
            .await
        }

        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama_chat/") => {
            crate::run::backend::openai::embed(
                crate::run::backend::openai::Provider::Ollama,
                &m[12..],
                &data,
            )
            .await
        }

        #[cfg(feature = "openai")]
        m if m.starts_with("openai/") => {
            crate::run::backend::openai::embed(
                crate::run::backend::openai::Provider::OpenAI,
                &m[7..],
                &data,
            )
            .await
        }

        #[cfg(feature = "gemini")]
        m if m.starts_with("gemini/") => {
            crate::run::backend::openai::embed(
                crate::run::backend::openai::Provider::Gemini,
                &m[7..],
                &data,
            )
            .await
        }

        _ => todo!("Unsupported embedding model {embedding_model}"),
    }
}
