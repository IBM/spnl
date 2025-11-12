use super::layer1::process_corpora;
use crate::{
    augment::AugmentOptions,
    ir::{Augment, Query},
};

/// Index by embedding only
pub async fn index(
    _query: &Query,
    a: &[(String, Augment)], // (enclosing_model, Augment)
    options: &AugmentOptions,
    m: &indicatif::MultiProgress,
) -> anyhow::Result<()> {
    let _ = process_corpora(a, options, m).await?;
    Ok(())
}
