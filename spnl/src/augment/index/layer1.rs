use anyhow::anyhow;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use super::windowing;

use crate::{
    Augment, Document,
    augment::{
        AugmentOptions,
        embed::{EmbedData, embed},
        storage,
    },
};

/// Number of fragment embeddings to perform in a single call
const BATCH_SIZE: usize = 64;

/// Fragment, embed, and index the corpora implied by the given
/// `Augment` structs
pub async fn process_corpora(
    a: &[Augment],
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<()> {
    let _ = futures::future::try_join_all(
        a.iter()
            .map(|augmentation| process_document(augmentation, options, m)),
    )
    .await?;

    Ok(())
}

/// Fragment, embed, and index the given document
async fn process_document(
    a: &Augment,
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<()> {
    let (filename, content) = &a.doc;
    let window_size = match content {
        Document::Text(_) => 1,
        Document::Binary(_) => 8,
    };

    let table_name = storage::VecDB::sanitize_table_name(
        format!(
            "{}.{}.{window_size}.{filename}",
            options.vecdb_table, a.embedding_model
        )
        .as_str(),
    );
    let done_file = ::std::path::PathBuf::from(&options.vecdb_uri).join(format!("{table_name}.ok"));

    if !::std::fs::exists(&done_file)? {
        let doc_content = match (
            content,
            ::std::path::Path::new(filename)
                .extension()
                .and_then(std::ffi::OsStr::to_str),
        ) {
            (Document::Text(content), Some("txt")) => windowing::text(content),
            (Document::Text(content), Some("jsonl")) => windowing::jsonl(content),
            (Document::Binary(content), Some("pdf")) => windowing::pdf(content, window_size),
            _ => Err(anyhow!("Unsupported `with` binary document {filename}")),
        }?;

        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:35.cyan/blue} {pos:>7}/{len:7} {msg}",
        )?;
        let pb = m.add(
            ProgressBar::new((doc_content.len() / BATCH_SIZE).try_into()?)
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
        for docs in doc_content.chunks(BATCH_SIZE) {
            let vecs = embed(&a.embedding_model, EmbedData::Vec(docs.to_vec()))
                .await?
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
        let db = storage::VecDB::connect(&options.vecdb_uri, table_name.as_str()).await?;
        db.add_vector(doc_content.as_slice(), docs_vectors, 1024)
            .await?;

        ::std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(done_file)?;
    }

    Ok(())
}
