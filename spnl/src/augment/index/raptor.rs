use super::layer1::layer1;

/// Index using the RAPTOR algorithm https://github.com/parthsarthi03/raptor
pub async fn index(
    a: &[crate::Augment],
    options: &crate::augment::AugmentOptions,
    m: &indicatif::MultiProgress,
) -> anyhow::Result<()> {
    layer1(a, options, m).await?;
    Ok(())
}
