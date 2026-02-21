use std::path::PathBuf;

// for .unique()
use itertools::Itertools;

// for shuffle support
use rand::seq::SliceRandom;

use crate::{
    augment::{
        AugmentOptions,
        embed::{EmbedData, embed},
    },
    ir::{Document, Query},
};

use leann_core::hnsw::{
    graph::VectorStorage,
    io::read_hnsw_index,
    search::{SearchParams, search_hnsw},
};
use leann_core::index::IndexPaths;
use leann_core::passages::{PassageManager, load_id_map};

/// Sanitize a name for use as a filesystem path component (mirrors layer1)
fn sanitize_name(name: &str) -> String {
    name.replace("/", "_").replace(":", "_")
}

/// Retrieve relevant document fragments using HNSW vector search
pub async fn retrieve(
    embedding_model: &String,
    body: &Query,
    (filename, _content): &(String, Document),
    options: &AugmentOptions,
) -> anyhow::Result<Vec<String>> {
    #[cfg(feature = "rag-deep-debug")]
    let verbose = ::std::env::var("SPNL_RAG_VERBOSE")
        .map(|var| !matches!(var.as_str(), "false"))
        .unwrap_or(false);

    #[cfg(feature = "rag-deep-debug")]
    let now = ::std::time::Instant::now();

    let max_matches: usize = options.max_aug.unwrap_or(10);

    // Derive the index path (must match layer1.rs naming)
    let index_name = sanitize_name(&format!(
        "default.{embedding_model}.{filename}.{:?}",
        options.indexer
    ));
    let index_dir = PathBuf::from(&options.index_dir);
    let index_path = index_dir.join(format!("{index_name}.leann"));
    let paths = IndexPaths::new(&index_path);

    // Load HNSW graph from disk
    let mut index_file = std::fs::File::open(paths.index_file_path())?;
    let graph = read_hnsw_index(&mut index_file)?;

    // Extract stored vectors
    let stored_vectors: Vec<f32> = match &graph.vector_storage {
        VectorStorage::Raw { data, .. } => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        VectorStorage::Null => {
            anyhow::bail!("HNSW index has no stored vectors; was it built with is_recompute=false?")
        }
    };

    // Load passage manager and ID map
    let passage_source = leann_core::index::PassageSource {
        source_type: "jsonl".to_string(),
        path: paths.passages_path().to_string_lossy().to_string(),
        index_path: paths.offset_path().to_string_lossy().to_string(),
        path_relative: None,
        index_path_relative: None,
    };
    let passages = PassageManager::load(&[passage_source], None)?;
    let id_map = load_id_map(&paths.id_map_path())?;

    #[cfg(feature = "rag-deep-debug")]
    if verbose {
        eprintln!("Embedding question {body}");
    }

    // Embed the query
    let body_vectors: Vec<Vec<f32>> = embed(embedding_model, EmbedData::Query(body.clone()))
        .await?
        .collect();

    #[cfg(feature = "rag-deep-debug")]
    if verbose {
        eprintln!("Matching question to document");
    }

    // Search for each query vector
    let params = SearchParams::default();
    let matching_ids: Vec<String> = body_vectors
        .into_iter()
        .flat_map(|query_vec| {
            let (labels, _distances) =
                search_hnsw(&graph, &query_vec, max_matches, &stored_vectors, &params);
            labels
                .into_iter()
                .filter_map(|label| id_map.get(label).cloned())
        })
        .unique()
        .collect();

    // Resolve passage text
    #[cfg(feature = "rag-deep-debug")]
    if verbose {
        eprintln!(
            "RAG fragments total_passages {} relevant_fragments {}",
            passages.len(),
            matching_ids.len()
        );
    }

    let mut d: Vec<String> = matching_ids
        .into_iter()
        .rev() // reverse so most relevant is closest to query (at end)
        .filter_map(|id| {
            passages
                .get_passage(&id)
                .ok()
                .map(|p| format!("Relevant Document {id}: {}", p.text))
        })
        .collect();

    #[cfg(feature = "rag-deep-debug")]
    if verbose {
        eprintln!("RAG time {:.2?} ms", now.elapsed().as_millis());
    }

    if options.shuffle {
        d.shuffle(&mut rand::rng());
    }
    Ok(d)
}
