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

/// Adapter that implements leann_core's EmbeddingProvider trait by
/// delegating to spnl's async embed() function via block_on.
pub struct SpnlEmbeddingProvider {
    pub model: String,
    pub dimensions: usize,
}

impl leann_core::embedding::EmbeddingProvider for SpnlEmbeddingProvider {
    fn compute_embeddings(&self, chunks: &[String]) -> anyhow::Result<ndarray::Array2<f32>> {
        // Run spnl's async embed() in a fresh tokio runtime on a
        // separate thread. This avoids deadlocking when called from
        // within std::thread::scope (as LeannBuilder::build_index does)
        // regardless of the outer tokio runtime flavor.
        let model = self.model.clone();
        let chunks_owned = chunks.to_vec();
        let vecs: Vec<Vec<f32>> = std::thread::scope(|s| {
            s.spawn(|| {
                tokio::runtime::Runtime::new()
                    .expect("Failed to create tokio runtime for embedding")
                    .block_on(embed(&model, EmbedData::Vec(chunks_owned)))
            })
            .join()
            .expect("Embedding thread panicked")
        })?
        .collect();

        let nrows = vecs.len();
        let ncols = self.dimensions;
        let mut data = Vec::with_capacity(nrows * ncols);
        for v in &vecs {
            if v.len() < ncols {
                data.extend_from_slice(v);
                data.resize(data.len() + ncols - v.len(), 0.0);
            } else {
                data.extend_from_slice(&v[..ncols]);
            }
        }

        Ok(ndarray::Array2::from_shape_vec((nrows, ncols), data)?)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        &self.model
    }
}
