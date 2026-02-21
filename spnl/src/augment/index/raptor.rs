use std::collections::HashMap;

use futures::{StreamExt, TryStreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use super::layer1::{IndexedCorpus, process_corpora};
use crate::augment::{AugmentOptions, embed::SpnlEmbeddingProvider};
use crate::{
    generate::{GenerateOptions, generate},
    ir::{Augment, Generate, GenerateMetadata, Message::*, Query, Repeat},
};

use leann_core::hnsw::{
    graph::VectorStorage,
    io::read_hnsw_index,
    search::{SearchParams, search_hnsw},
};
use leann_core::index::IndexPaths;
use leann_core::passages::{PassageManager, load_id_map};

/// Maximum concurrent calls to llm generate for summarization.
const CONCURRENCY_LIMIT: usize = 32;

/// Index using the RAPTOR algorithm https://github.com/parthsarthi03/raptor
///
/// This uses a two-phase approach because leann-rs's LeannBuilder only
/// supports bulk index builds (no incremental insertion):
///
/// Phase 1: Build initial HNSW index from base fragments
/// Phase 2: Search the initial index to find similar passages per
///          fragment, generate LLM summaries
/// Phase 3: Rebuild the HNSW index with originals + all summaries
///
/// TODO: This means summaries cannot influence each other during
/// cross-indexing (unlike the original lancedb approach which inserted
/// summaries incrementally). See RAPTOR_LEANN_PLAN.md for future work
/// on incremental HNSW insertion in leann-rs.
pub async fn index(
    query: &Query,
    a: &[(String, Augment)], // (enclosing_model, Augment)
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<()> {
    #[cfg(feature = "pull")]
    crate::pull::pull_if_needed(query).await?;

    // Phase 1: Build initial indexes
    let corpora = process_corpora(a, options, m).await?;

    // Phases 2+3: Cross-index each corpus
    for corpus in corpora {
        cross_index(corpus, options, m).await?;
    }

    Ok(())
}

/// Cross-index a single corpus:
/// Phase 2: Search initial index for similar passages, generate summaries
/// Phase 3: Rebuild index with originals + summaries
async fn cross_index(
    corpus: IndexedCorpus,
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<()> {
    let IndexedCorpus {
        filename,
        index_path,
        enclosing_model,
        embedding_model,
        dimensions,
    } = corpus;

    let file_base_name = ::std::path::Path::new(&filename)
        .file_name()
        .ok_or(anyhow::anyhow!("Could not determine base name"))?
        .display()
        .to_string();

    let paths = IndexPaths::new(&index_path);

    // Load the initial HNSW index for searching
    let mut index_file = std::fs::File::open(paths.index_file_path())?;
    let graph = read_hnsw_index(&mut index_file)?;

    let stored_vectors: Vec<f32> = match &graph.vector_storage {
        VectorStorage::Raw { data, .. } => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        VectorStorage::Null => {
            anyhow::bail!("HNSW index has no stored vectors for RAPTOR cross-indexing")
        }
    };

    // Load passages and ID map
    let passage_source = leann_core::index::PassageSource {
        source_type: "jsonl".to_string(),
        path: paths.passages_path().to_string_lossy().to_string(),
        index_path: paths.offset_path().to_string_lossy().to_string(),
        path_relative: None,
        index_path_relative: None,
    };
    let passage_mgr = PassageManager::load(&[passage_source], None)?;
    let id_map = load_id_map(&paths.id_map_path())?;

    let num_passages = id_map.len();
    let pb = m.add(
        ProgressBar::new(num_passages as u64)
            .with_style(ProgressStyle::with_template(
                "{msg} {wide_bar:.gray/green} {pos:>7}/{len:7} [{elapsed_precise}]",
            )?)
            .with_message(format!("Cross-indexing {file_base_name}")),
    );
    pb.tick();

    // Phase 2: For each passage, find similar passages and generate a summary
    let max_matches: usize = options.max_aug.unwrap_or(10);
    let params = SearchParams::default();

    let summary_futures = (0..num_passages).map(|idx| {
        let passage_id = &id_map[idx];
        let _passage_text = passage_mgr
            .get_passage(passage_id)
            .map(|p| p.text.clone())
            .unwrap_or_default();

        // Find similar passages using the stored vector for this passage
        let d = graph.dimensions;
        let start = idx * d;
        let end = start + d;
        let query_vec = &stored_vectors[start..end];

        let (labels, _distances) =
            search_hnsw(&graph, query_vec, max_matches, &stored_vectors, &params);

        let similar_texts: Vec<Query> = labels
            .into_iter()
            .filter_map(|label| {
                id_map.get(label).and_then(|id| {
                    passage_mgr
                        .get_passage(id)
                        .ok()
                        .map(|p| Query::Message(User(p.text.clone())))
                })
            })
            .collect();

        generate_summary(
            idx,
            file_base_name.clone(),
            similar_texts,
            enclosing_model.clone(),
            options,
            pb.clone(),
            m,
        )
    });

    // Execute summaries with bounded concurrency
    let summaries: Vec<(String, String)> = futures::stream::iter(summary_futures)
        .buffer_unordered(CONCURRENCY_LIMIT)
        .try_collect()
        .await?;

    pb.finish();

    // Phase 3: Rebuild the index with originals + summaries
    let rebuild_pb = m.add(
        ProgressBar::new(1)
            .with_style(ProgressStyle::with_template(
                "{msg} {wide_bar:.yellow/blue} {pos:>7}/{len:7} [{elapsed_precise}]",
            )?)
            .with_message(format!("Rebuilding index {file_base_name}")),
    );
    rebuild_pb.tick();

    let mut builder = leann_core::LeannBuilder::new(&embedding_model, Some(dimensions), "spnl")
        .with_recompute(false)
        .with_compact(false);

    // Re-add original passages
    for id in &id_map {
        if let Ok(p) = passage_mgr.get_passage(id) {
            let mut metadata = HashMap::new();
            metadata.insert("id".to_string(), serde_json::Value::String(id.clone()));
            builder.add_text(&p.text, metadata);
        }
    }

    // Add RAPTOR summaries
    for (id, summary_text) in &summaries {
        let mut metadata = HashMap::new();
        metadata.insert("id".to_string(), serde_json::Value::String(id.clone()));
        builder.add_text(summary_text, metadata);
    }

    // Rebuild
    let provider = SpnlEmbeddingProvider {
        model: embedding_model.clone(),
        dimensions,
    };
    builder.build_index(&index_path, &provider)?;

    rebuild_pb.finish();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn generate_summary(
    idx: usize,
    file_base_name: String,
    similar_texts: Vec<Query>,
    enclosing_model: String,
    options: &AugmentOptions,
    pb: ProgressBar,
    m: &MultiProgress,
) -> anyhow::Result<(String, String)> {
    let num_fragments = similar_texts.len().saturating_sub(1);
    let original_length = similar_texts
        .iter()
        .map(|q| match q {
            Query::Message(User(s)) => s.len(),
            _ => 0,
        })
        .sum::<usize>();

    let max_tokens = Some(100);
    let temperature = Some(0.2);

    let summary = match generate(
        Repeat {
            n: 1,
            generate: Generate {
                metadata: GenerateMetadata {
                    model: enclosing_model.to_string(),
                    max_tokens,
                    temperature,
                },
                input: Box::from(Query::Cross(vec![
                    Query::Message(System("You are a helpful assistant.".into())),
                    Query::Message(User(
                        "Write a summary of the following, including as many key details as possible:"
                            .into(),
                    )),
                    Query::Plus(similar_texts),
                ])),
            },
        },
        Some(m),
        &GenerateOptions::default(),
    )
    .await?
    {
        Query::Message(User(s)) => s,
        _ => "".into(),
    };

    if options.verbose {
        let summarized_length = summary.len();
        m.println(
            format!("Raptor summary fragments={num_fragments} original={original_length} summarized={summarized_length} \x1b[2m{summary}")
        )?;
    }

    pb.inc(1);
    Ok((format!("@raptor-{file_base_name}-{idx}"), summary))
}
