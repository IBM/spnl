use futures::future::try_join_all;
use indicatif::{MultiProgress, ProgressBar};

use crate::{
    Document, Query,
    run::{
        result::SpnlError,
        with::{
            embed::{EmbedData, embed},
            storage,
        },
    },
};

/// This fragments and windows the lines in the given PDF content. For
/// example if bytes="a\nb\nc\nd" and window_width=2, this will
/// produce ["a\nb", "b\nc", "c\nd"]
fn windowed_pdf(bytes: &[u8], window_width: usize) -> Result<Vec<String>, SpnlError> {
    Ok(pdf_extract::extract_text_from_mem(bytes)?
        .lines()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .windows(window_width)
        .step_by(2)
        .map(|s| s.join("\n"))
        .collect())
}

/// This treats every line of text as a separate document, with no
/// need for windowing or sub-fragmentation.
fn windowed_text(s: &str) -> Result<Vec<String>, SpnlError> {
    Ok(s.lines().map(|s| s.to_string()).collect())
}

#[derive(serde::Deserialize)]
struct JsonlText {
    text: String,
}

/// This treats every jsonl line as a separate document, with no need
/// for windowing or sub-fragmentation.
fn windowed_jsonl(s: &str) -> Result<Vec<String>, SpnlError> {
    Ok(serde_json::Deserializer::from_str(s)
        .into_iter::<JsonlText>()
        .filter_map(|line| match line {
            Ok(JsonlText { text }) => Some(text),
            Err(s) => {
                eprintln!("Error parsing jsonl line {s}");
                None
            }
        })
        .collect())
}

fn extract_augments(query: &Query) -> Vec<crate::Augment> {
    match query {
        Query::Generate(crate::Generate { input, .. }) => extract_augments(input),
        Query::Plus(v) | Query::Cross(v) => v.iter().flat_map(extract_augments).collect(),
        Query::Augment(a) => vec![a.clone()],
        _ => vec![],
    }
}

pub async fn run(query: &Query, db_uri: &str, table_name_base: &str) -> Result<(), SpnlError> {
    let m = MultiProgress::new();
    let _ = try_join_all(
        extract_augments(query)
            .into_iter()
            .map(|augmentation| index(augmentation, db_uri, table_name_base, &m)),
    )
    .await?;

    Ok(())
}

async fn index(
    a: crate::Augment,
    db_uri: &str,
    table_name_base: &str,
    m: &MultiProgress,
) -> Result<(), SpnlError> {
    let (filename, content) = &a.doc;
    let window_size = match content {
        Document::Text(_) => 1,
        Document::Binary(_) => 8,
    };

    let batch_size = 64; // Number of fragment embeddings to perform in a single call

    let table_name = storage::VecDB::sanitize_table_name(
        format!(
            "{table_name_base}.{}.{window_size}.{filename}",
            a.embedding_model
        )
        .as_str(),
    );
    let db = storage::VecDB::connect(db_uri, table_name.as_str()).await?;

    let done_file = ::std::path::PathBuf::from(db_uri).join(format!("{table_name}.ok"));
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
            _ => Err(Box::from(format!(
                "Unsupported `with` binary document {filename}"
            ))),
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
                        .ok_or("Could not determine base name")?
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
