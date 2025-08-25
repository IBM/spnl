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

pub struct Fragments {
    /// Name of corpus
    pub filename: String,

    /// Name of vector db table to use for subsequent indexing tasks
    pub table_name: String,

    /// The model to be used to generate over these fragments
    pub enclosing_model: String,

    /// The model to be used to generate over these fragments
    pub embedding_model: String,

    /// A list of vector embeddings
    pub fragments: Vec<Vec<f32>>,
}

/// Fragment, embed, and index the corpora implied by the given
/// `Augment` structs
pub async fn process_corpora(
    a: &[(String, Augment)], // (enclosing_model, Augment)
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<impl Iterator<Item = Fragments>> {
    Ok(futures::future::try_join_all(
        a.iter()
            .map(|augmentation| process_document(augmentation, options, m)),
    )
    .await?
    .into_iter()
    .flatten())
}

/// Fragment, embed, and index the given document
async fn process_document(
    (enclosing_model, a): &(String, Augment),
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<Option<Fragments>> {
    let (filename, content) = &a.doc;
    let window_size = match content {
        Document::Text(_) => 1,
        Document::Binary(_) => 8,
    };

    let file_base_name = ::std::path::Path::new(filename)
        .file_name()
        .ok_or(anyhow!("Could not determine base name"))?
        .display();
    let table_name = storage::VecDB::sanitize_table_name(
        format!(
            "{}.{}.{window_size}.{filename}.{:?}",
            options.vecdb_table, a.embedding_model, options.indexer,
        )
        .as_str(),
    );
    let done_file = ::std::path::PathBuf::from(&options.vecdb_uri).join(format!("{table_name}.ok"));

    if !::std::fs::exists(&done_file)? {
        // this is a list of fragment strings, i.e. we have broken up
        // the `content` into a list of fragments
        let fragments = match (
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

        let pb = m.add(
            ProgressBar::new(fragments.len().try_into()?)
                .with_style(ProgressStyle::with_template(
                    "{msg} {wide_bar:.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]",
                )?)
                .with_message(format!("Indexing {}", file_base_name,)),
        );

        // Needed to render the progress bar immediately (it will show
        // "0/N"). Otherwise, the progress bar would not even appear
        // until the first update.
        pb.inc(0);

        // This will be a parallel array (to `fragments`), containing
        // one vector embedding per fragment.
        let mut vector_embeddings = vec![];

        // For efficiency (confirm?) we ask for BATCH_SIZE embeddings at a time.
        for docs in fragments.chunks(BATCH_SIZE) {
            let n = docs.len();
            let vecs = embed(&a.embedding_model, EmbedData::Vec(docs.to_vec()))
                .await?
                .map(|vec| {
                    if vec.len() < 1024 {
                        // pad out to 1024
                        let mut ee = vec.clone();
                        ee.resize(1024, 0.0);
                        ee
                    } else {
                        vec
                    }
                });

            pb.inc(n.try_into()?);
            vector_embeddings.extend(vecs);
        }
        pb.finish();

        let db = storage::VecDB::connect(&options.vecdb_uri, table_name.as_str()).await?;

        if options.verbose {
            m.println(format!(
                "Inserting document embeddings {}",
                vector_embeddings.len()
            ))?;
        }

        // Recall that `fragments` and `vector_embeddings` have been
        // constructed as parallel arrays.
        db.add_vector(
            fragments
                .iter()
                .enumerate()
                .map(|(idx, fragment)| format!("@base-{}-{idx}: {fragment}", file_base_name)),
            vector_embeddings.clone(),
            1024,
        )
        .await?;

        // mark this filename as done
        ::std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(done_file)?;

        return Ok(Some(Fragments {
            filename: filename.clone(),
            table_name,
            enclosing_model: enclosing_model.clone(),
            embedding_model: a.embedding_model.clone(),
            fragments: vector_embeddings,
        }));
    }

    Ok(None)
}
