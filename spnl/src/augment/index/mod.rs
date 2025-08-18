mod layer1;
mod raptor;
mod simple_embed_retrieve;
mod windowing;

use indicatif::MultiProgress;

use crate::{
    Augment, Query,
    augment::{AugmentOptions, Indexer},
};

fn extract_augments(query: &Query) -> Vec<Augment> {
    match query {
        Query::Generate(crate::Generate { input, .. }) => extract_augments(input),
        Query::Plus(v) | Query::Cross(v) => v.iter().flat_map(extract_augments).collect(),
        Query::Augment(a) => vec![a.clone()],
        _ => vec![],
    }
}

pub async fn index(query: &Query, options: &AugmentOptions) -> anyhow::Result<()> {
    let m = MultiProgress::new();
    let augments = extract_augments(query);

    match options.indexer {
        Indexer::Raptor => raptor::index(&augments, options, &m).await,
        Indexer::SimpleEmbedRetrieve => simple_embed_retrieve::index(&augments, options, &m).await,
    }
}
