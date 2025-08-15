use anyhow::anyhow;
use indicatif::{MultiProgress, ProgressBar};

mod windowing;
pub use windowing::*;

use crate::{
    Document, Query,
    augment::{
        AugmentOptions,
        embed::{EmbedData, embed},
        storage,
    },
};

fn extract_augments(query: &Query) -> Vec<crate::Augment> {
    match query {
        Query::Generate(crate::Generate { input, .. }) => extract_augments(input),
        Query::Plus(v) | Query::Cross(v) => v.iter().flat_map(extract_augments).collect(),
        Query::Augment(a) => vec![a.clone()],
        _ => vec![],
    }
}

pub async fn index(query: &Query, options: &AugmentOptions) -> anyhow::Result<()> {
    let m = MultiProgress::new();
    let _ = futures::future::try_join_all(
        extract_augments(query)
            .into_iter()
            .map(|augmentation| index_document(augmentation, options, &m)),
    )
    .await?;

    Ok(())
}

async fn index_document(
    a: crate::Augment,
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<()> {
    let (filename, content) = &a.doc;
    let window_size = match content {
        Document::Text(_) => 1,
        Document::Binary(_) => 8,
    };

    let batch_size = 64; // Number of fragment embeddings to perform in a single call

    let table_name = storage::VecDB::sanitize_table_name(
        format!(
            "{}.{}.{window_size}.{filename}",
            options.vecdb_table, a.embedding_model
        )
        .as_str(),
    );
    let db = storage::VecDB::connect(&options.vecdb_uri, table_name.as_str()).await?;

    let done_file = ::std::path::PathBuf::from(&options.vecdb_uri).join(format!("{table_name}.ok"));
    if !::std::fs::exists(&done_file)? {
        let doc_content = match (
            content,
            ::std::path::Path::new(filename)
                .extension()
                .and_then(std::ffi::OsStr::to_str),
        ) {
            (Document::Text(content), Some("txt")) => windowed_text(content),
            (Document::Text(content), Some("jsonl")) => windowed_jsonl(content),
            (Document::Binary(content), Some("pdf")) => windowed_pdf(content, window_size),
            _ => Err(anyhow!("Unsupported `with` binary document {filename}")),
        }?;
        let key = doc_content.as_slice();

        let sty = indicatif::ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:35.cyan/blue} {pos:>7}/{len:7} {msg}",
        )?;
        let pb = m.add(
            ProgressBar::new((doc_content.len() / batch_size).try_into()?)
                .with_style(sty)
                .with_message(
                    ::std::path::Path::new(filename)
                        .file_name()
                        .ok_or(anyhow!("Could not determine base name"))?
                        .display()
                        .to_string(),
                ),
        );
        pb.inc(0);
        let mut docs_vectors = vec![];
        for docs in doc_content.chunks(batch_size) {
            let vecs = embed(&a.embedding_model, EmbedData::Vec(docs.to_vec()))
                .await?
                .into_iter()
                .map(|vec| {
                    if vec.len() < 1024 {
                        let mut ee = vec.clone();
                        ee.resize(1024, 0.0);
                        ee
                    } else {
                        vec
                    }
                });
            pb.inc(1);
            docs_vectors.extend(vecs);
        }
        pb.finish();

        eprintln!("Inserting document embeddings {}", docs_vectors.len());
        db.add_vector(key, docs_vectors, 1024).await?;

        ::std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(done_file)?;
    }

    Ok(())
}
