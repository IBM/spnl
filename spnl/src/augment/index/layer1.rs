use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::anyhow;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::{
    augment::{AugmentOptions, embed::SpnlEmbeddingProvider},
    ir::{Augment, Document},
};

/// Sanitize a name for use as a filesystem path component
fn sanitize_name(name: &str) -> String {
    name.replace("/", "_").replace(":", "_")
}

/// Fragment, embed, and index the corpora implied by the given
/// `Augment` structs
pub async fn process_corpora(
    a: &[(String, Augment)], // (enclosing_model, Augment)
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<Vec<IndexedCorpus>> {
    // Process documents sequentially since LeannBuilder uses
    // block_in_place internally for embedding computation
    let mut results = Vec::new();
    for augmentation in a {
        if let Some(corpus) = process_document(augmentation, options, m).await? {
            results.push(corpus);
        }
    }
    Ok(results)
}

/// Metadata about an indexed corpus, used by RAPTOR for cross-indexing
pub struct IndexedCorpus {
    /// Name of corpus
    pub filename: String,

    /// The leann index path (e.g. data/spnl/name.leann)
    pub index_path: PathBuf,

    /// The model to be used to generate over these fragments
    pub enclosing_model: String,

    /// The embedding model used
    pub embedding_model: String,

    /// Embedding dimensions
    pub dimensions: usize,
}

/// Fragment, embed, and index the given document
async fn process_document(
    (enclosing_model, a): &(String, Augment),
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<Option<IndexedCorpus>> {
    #[cfg(feature = "pull")]
    crate::pull::pull_model_if_needed(a.embedding_model.as_str()).await?;

    let (filename, content) = &a.doc;

    let file_base_name = ::std::path::Path::new(filename)
        .file_name()
        .ok_or(anyhow!("Could not determine base name"))?
        .display()
        .to_string();

    let index_name = sanitize_name(&format!(
        "default.{}.{filename}.{:?}",
        a.embedding_model, options.indexer,
    ));
    let index_dir = PathBuf::from(&options.index_dir);
    let index_path = index_dir.join(format!("{index_name}.leann"));
    let done_file = index_dir.join(format!("{index_name}.ok"));

    if ::std::fs::exists(&done_file)? {
        // Detect dimensions from existing index metadata
        let meta = leann_core::IndexMeta::load(
            &leann_core::index::IndexPaths::new(&index_path).meta_path(),
        )?;
        return Ok(Some(IndexedCorpus {
            filename: filename.clone(),
            index_path,
            enclosing_model: enclosing_model.clone(),
            embedding_model: a.embedding_model.clone(),
            dimensions: meta.dimensions,
        }));
    }

    // Extract text from document
    let text = match (
        content,
        ::std::path::Path::new(filename)
            .extension()
            .and_then(std::ffi::OsStr::to_str),
    ) {
        (Document::Text(content), _) => content.clone(),
        (Document::Binary(content), Some("pdf")) => pdf_extract::extract_text_from_mem(content)?,
        _ => return Err(anyhow!("Unsupported `index` document type: {filename}")),
    };

    // Chunk with leann-rs sentence-based chunking
    let chunks = leann_core::chunking::chunk_text(&text, options.chunk_size, options.chunk_overlap);
    if chunks.is_empty() {
        return Err(anyhow!("No chunks produced from document: {filename}"));
    }

    let pb = m.add(
        ProgressBar::new(chunks.len().try_into()?)
            .with_style(ProgressStyle::with_template(
                "{msg} {wide_bar:.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]",
            )?)
            .with_message(format!("Indexing {file_base_name}")),
    );
    pb.inc(0);

    // Detect embedding dimensions via a probe embedding
    let probe = crate::augment::embed::embed(
        &a.embedding_model,
        crate::augment::embed::EmbedData::String("probe".to_string()),
    )
    .await?
    .next()
    .ok_or_else(|| anyhow!("Embedding model returned no vectors for probe"))?;
    let dimensions = probe.len();

    // Create LeannBuilder with stored vectors (is_recompute=false) so
    // we can search without a ZMQ embedding server at retrieve time.
    // is_compact=false because CSR conversion prunes stored vectors.
    let mut builder = leann_core::LeannBuilder::new(&a.embedding_model, Some(dimensions), "spnl")
        .with_recompute(false)
        .with_compact(false);

    // Add chunks as passages with @base-{name}-{idx} IDs
    for (idx, chunk) in chunks.iter().enumerate() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "id".to_string(),
            serde_json::Value::String(format!("@base-{file_base_name}-{idx}")),
        );
        builder.add_text(chunk, metadata);
    }

    pb.set_length(chunks.len() as u64);

    // Ensure index directory exists
    ::std::fs::create_dir_all(&index_dir)?;

    // Build the HNSW index
    let provider = SpnlEmbeddingProvider {
        model: a.embedding_model.clone(),
        dimensions,
    };
    builder.build_index(&index_path, &provider)?;

    pb.finish();

    // Mark as done
    ::std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&done_file)?;

    Ok(Some(IndexedCorpus {
        filename: filename.clone(),
        index_path,
        enclosing_model: enclosing_model.clone(),
        embedding_model: a.embedding_model.clone(),
        dimensions,
    }))
}
