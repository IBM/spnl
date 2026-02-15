//! Embedding support for mistral.rs backend

use crate::augment::embed::{EmbedData, contentify};
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest, best_device};

/// Generate embeddings using mistral.rs backend
///
/// Note: Unlike text generation models, embedding models are loaded fresh each time
/// because the EmbeddingModel type is not publicly exported from mistralrs.
pub async fn embed(embedding_model: &str, data: &EmbedData) -> anyhow::Result<Vec<Vec<f32>>> {
    // Load the embedding model
    let device = best_device(false).expect("Failed to detect device");
    let model = EmbeddingModelBuilder::new(embedding_model)
        .with_device(device)
        .build()
        .await?;

    // Convert data to text strings
    let docs = match data {
        EmbedData::String(s) => vec![s.clone()],
        EmbedData::Vec(v) => v.clone(),
        EmbedData::Query(u) => contentify(u),
    };

    // Create an embedding request using the builder pattern
    let mut request = EmbeddingRequest::builder();
    for doc in docs {
        request = request.add_prompt(doc);
    }

    // Get embeddings from the model - returns Vec<Vec<f32>> directly
    let embeddings = model.generate_embeddings(request).await?;

    Ok(embeddings)
}

// Made with Bob
