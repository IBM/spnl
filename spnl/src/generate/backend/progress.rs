use crate::ir::GenerateMetadata;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub fn bars(
    n: usize,
    metadata: &GenerateMetadata,
    m: &Option<&MultiProgress>,
) -> anyhow::Result<Option<Vec<ProgressBar>>> {
    let style = ProgressStyle::with_template(
        "{msg} {wide_bar:.yellow/orange} {pos:>7}/{len:7} [{elapsed_precise}]",
    )?;

    Ok(m.map(|m| {
        ::std::iter::repeat_n(0, n)
            .enumerate()
            .map(|(idx, _)| {
                m.add(
                    metadata
                        .max_tokens
                        .map(|max_tokens| ProgressBar::new((max_tokens as u64) * 4))
                        .unwrap_or_else(ProgressBar::no_length)
                        .with_style(style.clone())
                        .with_message(if n == 1 {
                            "Generating".to_string()
                        } else {
                            format!("Bulk Generation ({})", idx + 1)
                        }),
                )
            })
            .collect()
    }))
}
