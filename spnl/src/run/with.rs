use futures::future::try_join_all;
use itertools::Itertools;
use sha2::Digest;

use crate::{
    Document, Query,
    run::{
        result::SpnlResult,
        with::embed::{EmbedData, embed},
    },
};

pub mod embed;
pub mod index;
mod storage;

pub async fn retrieve(
    embedding_model: &String,
    body: &Query,
    (filename, content): &(String, Document),
    db_uri: &str,
    table_name_base: &str,
) -> SpnlResult {
    let verbose = ::std::env::var("SPNL_RAG_VERBOSE")
        .map(|var| !matches!(var.as_str(), "false"))
        .unwrap_or(false);

    use std::time::Instant;
    let now = Instant::now();
    let max_matches = 100; // Maximum number of relevant fragments to consider

    let window_size = match content {
        Document::Text(_) => 1,
        Document::Binary(_) => 8,
    };

    let table_name = storage::VecDB::sanitize_table_name(
        format!("{table_name_base}.{embedding_model}.{window_size}.{filename}").as_str(),
    );
    let db = storage::VecDB::connect(db_uri, table_name.as_str()).await?;

    if verbose {
        eprintln!("Embedding question {body}");
    }
    let body_vectors = embed(embedding_model, EmbedData::Query(body.clone()))
        .await?
        .into_iter()
        .map(|v| {
            if v.len() < 1024 {
                let mut vv = v.clone();
                vv.resize(1024, 0.0);
                vv
            } else {
                v
            }
        })
        .collect::<Vec<_>>();

    if verbose {
        eprintln!("Matching question to document");
    }
    let matching_docs = try_join_all(
        body_vectors
            .into_iter()
            .map(|v| db.find_similar(v, max_matches)),
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

    if verbose {
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
    }

    let len1 = match content {
        Document::Text(c) => c.len(),
        Document::Binary(b) => b.len(),
    } as f64;
    let len2 = matching_docs.clone().map(|doc| doc.len()).sum::<usize>() as f64;
    if verbose {
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
        .enumerate()
        .map(|(idx, doc)| Query::User(format!("Relevant document {idx}: {doc}")))
        .collect::<Vec<_>>();

    if verbose {
        eprintln!("RAG time {:.2?} ms", now.elapsed().as_millis());
    }
    Ok(Query::Plus(d))
}
