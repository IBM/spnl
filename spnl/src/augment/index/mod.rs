mod layer1;
mod raptor;
mod simple_embed_retrieve;
pub mod windowing;

use indicatif::MultiProgress;

use crate::{
    augment::{AugmentOptions, Indexer},
    ir::{Augment, Generate, GenerateMetadata, Query},
};

fn extract_augments(query: &Query, enclosing_model: &Option<String>) -> Vec<(String, Augment)> {
    match (query, enclosing_model) {
        (
            Query::Generate(Generate {
                input,
                metadata: GenerateMetadata { model, .. },
            }),
            _,
        ) => extract_augments(input, &Some(model.clone())),
        (Query::Plus(v) | Query::Cross(v), _) => v
            .iter()
            .flat_map(|q| extract_augments(q, enclosing_model))
            .collect(),
        (Query::Augment(a), Some(enclosing_model)) => vec![(enclosing_model.clone(), a.clone())],
        _ => vec![],
    }
}

pub async fn index(query: &Query, options: &AugmentOptions) -> anyhow::Result<()> {
    let m = MultiProgress::new();
    let augments = extract_augments(query, &None);

    match options.indexer {
        Indexer::Raptor => raptor::index(&augments, options, &m).await,
        Indexer::SimpleEmbedRetrieve => {
            simple_embed_retrieve::index(query, &augments, options, &m).await
        }
    }
}
