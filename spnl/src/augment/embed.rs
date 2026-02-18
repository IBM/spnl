use crate::generate::backend::openai;
use crate::ir::{Message::*, Query};

pub enum EmbedData {
    String(String),
    Query(Query),
    Vec(Vec<String>),
}

/// Helper function to convert Query to text content for embeddings
pub fn contentify(input: &Query) -> Vec<String> {
    match input {
        Query::Seq(v) | Query::Plus(v) | Query::Cross(v) => v.iter().flat_map(contentify).collect(),
        Query::Message(Assistant(s)) | Query::Message(System(s)) => vec![s.clone()],
        o => {
            let s = o.to_string();
            if s.is_empty() {
                vec![]
            } else {
                vec![o.to_string()]
            }
        }
    }
}

pub async fn embed(
    embedding_model: &String,
    data: EmbedData,
) -> anyhow::Result<impl Iterator<Item = Vec<f32>>> {
    let embeddings: Vec<Vec<f32>> = match embedding_model {
        #[cfg(feature = "local")]
        m if m.starts_with("local/") => {
            crate::generate::backend::mistralrs::embed::embed(&m[6..], &data).await?
        }

        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama/") => openai::embed(openai::Provider::Ollama, &m[7..], &data)
            .await?
            .collect(),

        #[cfg(feature = "ollama")]
        m if m.starts_with("ollama_chat/") => {
            openai::embed(openai::Provider::Ollama, &m[12..], &data)
                .await?
                .collect()
        }

        #[cfg(feature = "openai")]
        m if m.starts_with("openai/") => openai::embed(openai::Provider::OpenAI, &m[7..], &data)
            .await?
            .collect(),

        #[cfg(feature = "gemini")]
        m if m.starts_with("gemini/") => openai::embed(openai::Provider::Gemini, &m[7..], &data)
            .await?
            .collect(),

        _ => todo!("Unsupported embedding model {embedding_model}"),
    };

    Ok(embeddings.into_iter())
}
