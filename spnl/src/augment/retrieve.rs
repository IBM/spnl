// for .unique()
use itertools::Itertools;

use crate::{
    Document, Query,
    augment::{
        AugmentOptions,
        embed::{EmbedData, embed},
        storage,
    },
};

/// Retrieve relevant
pub async fn retrieve(
    embedding_model: &String,
    body: &Query,
    (filename, content): &(String, Document),
    options: &AugmentOptions,
) -> anyhow::Result<impl Iterator<Item = String>> {
    #[cfg(feature = "rag_deep_debug")]
    let verbose = ::std::env::var("SPNL_RAG_VERBOSE")
        .map(|var| !matches!(var.as_str(), "false"))
        .unwrap_or(false);

    #[cfg(feature = "rag_deep_debug")]
    let now = ::std::time::Instant::now();

    // Maximum number of relevant fragments to consider
    let max_matches: usize = options.max_aug.unwrap_or(10);

    let window_size = match content {
        Document::Text(_) => 1,
        Document::Binary(_) => 8,
    };

    let table_name = storage::VecDB::sanitize_table_name(
        format!(
            "{}.{embedding_model}.{window_size}.{filename}.{:?}",
            options.vecdb_table, options.indexer
        )
        .as_str(),
    );
    let db = storage::VecDB::connect(&options.vecdb_uri, table_name.as_str()).await?;

    #[cfg(feature = "rag_deep_debug")]
    if verbose {
        eprintln!("Embedding question {body}");
    }

    let body_vectors = embed(embedding_model, EmbedData::Query(body.clone()))
        .await?
        .map(|v| {
            if v.len() < 1024 {
                let mut vv = v.clone();
                vv.resize(1024, 0.0);
                vv
            } else {
                v
            }
        });

    #[cfg(feature = "rag_deep_debug")]
    if verbose {
        eprintln!("Matching question to document");
    }

    // TODO find a way to use db.find_similar_keys() to avoid
    // replicating the filter_map logic below. Blocker: `db` is shared
    // across the asyncs because find_similar_keys() returns an
    // Iterator whereas find_similar() returns a vector.
    let matching_docs = futures::future::try_join_all(
        body_vectors
            .into_iter()
            .map(|v| db.find_similar(v, max_matches, None, None)),
    )
    .await?
    .into_iter()
    .flatten()
    .filter_map(|record_batch| {
        if let Some(files_array) = record_batch.column_by_name("filename")
            && let Some(files) = files_array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
        {
            return Some(
                files
                    .iter()
                    .filter_map(|b| b.map(|b| b.to_string()))
                    .collect::<Vec<String>>(),
            );
        }

        // no matching docs for this body vector
        None
    })
    .flatten()
    .unique();

    #[cfg(feature = "rag_deep_debug")]
    if verbose {
        use sha2::Digest;
        eprintln!(
            "RAGSizes {}",
            matching_docs.clone().map(|doc| doc.len()).join(" ")
        );
        eprintln!(
            "RAGHashes {}",
            matching_docs
                .clone()
                .map(|doc| {
                    let mut hasher = sha2::Sha256::new();
                    hasher.update(doc);
                    format!("{:x}", hasher.finalize())
                })
                .join(" ")
        );

        let len1 = match content {
            Document::Text(c) => c.len(),
            Document::Binary(b) => b.len(),
        } as f64;
        let len2 = matching_docs.clone().map(|doc| doc.len()).sum::<usize>() as f64;
        eprintln!(
            "RAG fragments total_fragments {} relevant_fragments {}",
            match content {
                Document::Text(t) => t.len(),
                Document::Binary(b) => b.len(),
            },
            matching_docs.clone().count()
        );
        eprintln!(
            "RAG size reduction factor {:.2} {len1} -> {len2} bytes",
            len1 / len2,
        );
    }

    let d = matching_docs
        .into_iter()
        .rev() // reverse so that we can present the most relevant closest to the query (at the end)
        .map(|doc| format!("Relevant Document {doc}"));

    #[cfg(feature = "rag_deep_debug")]
    if verbose {
        eprintln!("RAG time {:.2?} ms", now.elapsed().as_millis());
    }

    Ok(d)
}
