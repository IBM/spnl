use crate::Query;
use crate::generate::backend::openai;

pub enum EmbedData {
    String(String),
    Query(Query),
    Vec(Vec<String>),
}

pub async fn embed(
    embedding_model: &String,
    data: EmbedData,
) -> anyhow::Result<impl Iterator<Item = Vec<f32>>> {
    match embedding_model {
        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama/") => {
            openai::embed(openai::Provider::Ollama, &m[7..], &data).await
        }

        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama_chat/") => {
            openai::embed(openai::Provider::Ollama, &m[12..], &data).await
        }

        #[cfg(feature = "openai")]
        m if m.starts_with("openai/") => {
            openai::embed(openai::Provider::OpenAI, &m[7..], &data).await
        }

        #[cfg(feature = "gemini")]
        m if m.starts_with("gemini/") => {
            openai::embed(openai::Provider::Gemini, &m[7..], &data).await
        }

        _ => todo!("Unsupported embedding model {embedding_model}"),
    }
}
