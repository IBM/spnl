mod layer1;
mod raptor;
mod simple_embed_retrieve;
mod windowing;

use indicatif::MultiProgress;

use crate::{
    Augment, Query,
    augment::{AugmentOptions, Indexer},
};

fn extract_augments(query: &Query) -> Box<dyn Iterator<Item = Augment> + '_> {
    match query {
        Query::Generate(crate::Generate { input, .. }) => Box::new(extract_augments(input)),
        Query::Plus(v) | Query::Cross(v) => Box::new(v.iter().flat_map(extract_augments)),
        Query::Augment(a) => Box::new(std::iter::once(a.clone())),
        _ => Box::new(std::iter::empty()),
    }
}

pub async fn index(query: &Query, options: &AugmentOptions) -> anyhow::Result<()> {
    let m = MultiProgress::new();
    let augments: Vec<_> = extract_augments(query).collect();

    match options.indexer {
        Indexer::Raptor => raptor::index(&augments, options, &m).await,
        Indexer::SimpleEmbedRetrieve => simple_embed_retrieve::index(&augments, options, &m).await,
    }
}
