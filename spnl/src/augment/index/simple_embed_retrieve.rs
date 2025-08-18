use super::layer1::process_corpora;

/// Index by embedding only
pub async fn index(
    a: &[crate::Augment],
    options: &crate::augment::AugmentOptions,
    m: &indicatif::MultiProgress,
) -> anyhow::Result<()> {
    process_corpora(a, options, m).await
}
