use futures::{StreamExt, TryStreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use itertools::Itertools; // for .unique() // for .buffer_unordered() and try_concat(), respectively

use super::layer1::{Fragments, process_corpora};
use crate::augment::storage;
use crate::{
    Augment, Query,
    augment::{
        AugmentOptions,
        embed::{EmbedData, embed},
    },
    generate::generate,
};

// TODO how do we determine the best concurrency limit?
const CONCURRENCY_LIMIT: usize = 32;

/// Index using the RAPTOR algorithm https://github.com/parthsarthi03/raptor
pub async fn index(
    a: &[(String, Augment)], // (enclosing_model, Augment)
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<()> {
    let futures = process_corpora(a, options, m) // this will generate one Fragments struct per corpus
        .await?
        .map(|f| cross_index(f, options, m)); // then iterate over each Fragments struct to "cross index" it

    // Create a buffered stream that will execute up to N futures in parallel
    // (without preserving the order of the results)
    let mut stream = futures::stream::iter(futures).buffer_unordered(CONCURRENCY_LIMIT);
    while (stream.try_next().await?).is_some() {} // TODO there must be a better way of doing this?

    Ok(())
}

/// "Cross index" the given `Fragments`, meaning, for each fragment in Fragments:
/// 1. Look it up (in the vector database `db`) to find similar fragments.
/// 2. Ask an LLM to summarize these related fragments.
/// 3. Embed that summary
/// 4. Insert summary+embedding into `db` to aid subsequent retrievals
async fn cross_index(
    Fragments {
        embedding_model,
        enclosing_model,
        fragments,
        filename,
        table_name,
        ..
    }: Fragments,
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<()> {
    let file_base_name = ::std::path::Path::new(&filename)
        .file_name()
        .ok_or(anyhow::anyhow!("Could not determine base name"))?
        .display();
    let db = storage::VecDB::connect(&options.vecdb_uri, table_name.as_str()).await?;
    let pb = m.add(
        ProgressBar::new(fragments.len() as u64)
            .with_style(ProgressStyle::with_template(
                "{msg} {wide_bar:.gray/green} {pos:>7}/{len:7} [{elapsed_precise}]",
            )?)
            .with_message(format!("Cross-indexing {}", file_base_name,)),
    );
    pb.tick(); // to get it to show up right away

    let futures = fragments.into_iter().enumerate().map(|(idx, f)| {
        cross_index_fragment(
            idx,
            file_base_name.to_string(),
            f,
            &embedding_model,
            &enclosing_model,
            &db,
            options,
            &pb,
            m,
        )
    });

    // Create a buffered stream that will execute up to N futures in parallel
    // (without preserving the order of the results)
    let mut stream = futures::stream::iter(futures).buffer_unordered(CONCURRENCY_LIMIT);
    while (stream.try_next().await?).is_some() {} // TODO there must be a better way of doing this?

    Ok(())
}

// TODO fix this allow
#[allow(clippy::too_many_arguments)]
async fn cross_index_fragment(
    idx: usize,
    file_base_name: String,
    fragment: (String, Vec<f32>),
    embedding_model: &String,
    enclosing_model: &str,
    db: &storage::VecDB,
    options: &AugmentOptions,
    pb: &ProgressBar,
    m: &MultiProgress,
) -> anyhow::Result<()> {
    // Maximum number of relevant fragments to consider
    let max_matches: usize = options.max_aug.unwrap_or(10);

    let re = ::regex::Regex::new("^@base.+: ")?;

    // TODO, this shares logic with retrieve.rs
    let input = db
        .find_similar(fragment.1, max_matches)
        .await?
        .into_iter()
        .filter_map(|record_batch| {
            if let Some(files_array) = record_batch.column_by_name("filename")
                && let Some(files) = files_array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
            {
                // Here are the fragments that are near to the given fragment
                Some(
                    files
                        .iter()
                        .filter_map(|b| b.map(|b| b.to_string()))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            }
        })
        .flatten()
        .unique()
        .map(|s| Query::User(re.replace(&s, "").to_string()))
        //.chain([Query::User(format!("Main Document: {}", fragment.0))])
        .collect::<Vec<_>>();

    let num_fragments = input.len() - 1;
    let original_length = input
        .iter()
        .map(|q| match q {
            Query::User(s) => s.len(),
            _ => 0,
        })
        .sum::<usize>();

    // TODO: hard-coded
    let max_tokens = &Some(100);
    let temp = &Some(0.2);

    let summary = match generate(
        enclosing_model,
        &Query::Cross(vec![
            //Query::System("You create concise summaries by extracting key concepts and term definitions".into()),
            Query::System("You are a helpful assistant.".into()), // copied from raptor python code
            Query::User(
                "Write a summary of the following, including as many key details as possible:"
                    .into(),
            ), // copied from raptor python code
            Query::Plus(input),
        ]),
        max_tokens,
        temp,
        Some(m),
        false,
    )
    .await?
    {
        Query::User(s) => s,
        _ => "".into(),
    };

    if options.verbose {
        let summarized_length = summary.len();
        m.println(
            format!("Raptor summary fragments={num_fragments} original={original_length} summarized={summarized_length} \x1b[2m{summary}")
        )?;
    }

    // Now embed and then insert the summary into the vector db (TODO?
    // batch these up?)
    let vector_embedding = embed(embedding_model, EmbedData::String(summary.clone())).await?;
    db.add_vector(
        [format!("@raptor-{file_base_name}-{idx}: {summary}")],
        vector_embedding,
        1024,
    )
    .await?;

    pb.inc(1);
    Ok(())
}
