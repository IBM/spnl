use super::layer1::process_corpora;

/// Index using the RAPTOR algorithm https://github.com/parthsarthi03/raptor
pub async fn index(
    a: &[crate::Augment],
    options: &crate::augment::AugmentOptions,
    m: &indicatif::MultiProgress,
) -> anyhow::Result<()> {
    // first, index the documents
    process_corpora(a, options, m).await?;

    Ok(())
}
