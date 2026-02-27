//! Embedding support for mistral.rs backend

use crate::augment::embed::{EmbedData, contentify};
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest, best_device};

/// Returns true if MISTRALRS_VERBOSE env var is set to "true" or "1"
fn should_enable_logging() -> bool {
    std::env::var("MISTRALRS_VERBOSE")
        .map(|v| v.to_lowercase() == "true" || v == "1")
        .unwrap_or(false)
}

/// Generate embeddings using mistral.rs backend
///
/// Note: Unlike text generation models, embedding models are loaded fresh each time
/// because the EmbeddingModel type is not publicly exported from mistralrs.
pub async fn embed(embedding_model: &str, data: &EmbedData) -> anyhow::Result<Vec<Vec<f32>>> {
    // Load the embedding model
    let device = best_device(false).expect("Failed to detect device");

    if should_enable_logging() {
        eprintln!(
            "Loading embedding model: {} on device: {:?}",
            embedding_model, device
        );
    }

    let mut builder = EmbeddingModelBuilder::new(embedding_model).with_device(device);

    // Optionally enable logging
    if should_enable_logging() {
        builder = builder.with_logging();
    }

    let model = builder.build().await?;

    if should_enable_logging() {
        eprintln!("Embedding model loaded successfully");
    }

    // Convert data to text strings
    let docs = match data {
        EmbedData::String(s) => vec![s.clone()],
        EmbedData::Vec(v) => v.clone(),
        EmbedData::Query(u) => contentify(u),
    };

    if should_enable_logging() {
        eprintln!("Generating embeddings for {} documents", docs.len());
    }

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
