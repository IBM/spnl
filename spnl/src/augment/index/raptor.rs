use futures::{StreamExt, TryStreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use super::layer1::{Fragments, process_corpora};
use crate::augment::storage;
use crate::{
    Augment,
    Message::*,
    Query,
    augment::{
        AugmentOptions,
        embed::{EmbedData, embed},
    },
    generate::generate,
};

/// Maximum concurrent calls to llm generate for summarization.
// TODO how do we determine the best concurrency limit?
const CONCURRENCY_LIMIT: usize = 32;

/// Index using the RAPTOR algorithm https://github.com/parthsarthi03/raptor
pub async fn index(
    query: &Query,
    a: &[(String, Augment)], // (enclosing_model, Augment)
    options: &AugmentOptions,
    m: &MultiProgress,
) -> anyhow::Result<()> {
    // TODO if we really want the pulls to be done in parallel with
    // the process_corpora, we'll need something fancier...
    #[cfg(feature = "pull")]
    crate::pull::pull_if_needed(query).await?;

    // This will generate one Fragments struct per corpus, and then iterate over each Fragments struct to "cross index" it
    let cross_index_futures = process_corpora(a, options, m)
        .await?
        .map(|f| cross_index(f, options, m));

    // Create a buffered stream that will execute up to N futures in parallel
    // (without preserving the order of the results)
    let mut stream = futures::stream::iter(cross_index_futures).buffer_unordered(CONCURRENCY_LIMIT);
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
    let pb = m.add(
        ProgressBar::new(fragments.len() as u64)
            .with_style(ProgressStyle::with_template(
                "{msg} {wide_bar:.gray/green} {pos:>7}/{len:7} [{elapsed_precise}]",
            )?)
            .with_message(format!("Cross-indexing {}", file_base_name,)),
    );
    pb.tick(); // to get it to show up right away

    let db = storage::VecDB::connect(&options.vecdb_uri, table_name.as_str()).await?;
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
    fragment: Vec<f32>,
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
        .find_similar_keys("filename", fragment, max_matches, None, None)
        .await?
        // .filter(|s| *s != fragment.0) // don't raptor-ize the very fragment we are tryign to summarize
        .map(|s| Query::Message(User(re.replace(&s, "").to_string())))
        .collect::<Vec<_>>();

    let num_fragments = input.len() - 1;
    let original_length = input
        .iter()
        .map(|q| match q {
            Query::Message(User(s)) => s.len(),
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
            Query::Message(System("You are a helpful assistant.".into())), // copied from raptor python code
            Query::Message(User(
                "Write a summary of the following, including as many key details as possible:"
                    .into(),
            )), // copied from raptor python code
            Query::Plus(input),
        ]),
        max_tokens,
        temp,
        Some(m),
        false,
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
