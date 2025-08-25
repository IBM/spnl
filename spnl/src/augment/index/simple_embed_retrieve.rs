use super::layer1::process_corpora;

/// Index by embedding only
pub async fn index(
    _query: &crate::Query,
    a: &[(String, crate::Augment)], // (enclosing_model, Augment)
    options: &crate::augment::AugmentOptions,
    m: &indicatif::MultiProgress,
) -> anyhow::Result<()> {
    let _ = process_corpora(a, options, m).await?;
    Ok(())
}
