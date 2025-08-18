/// This fragments and windows the lines in the given PDF content. For
/// example if bytes="a\nb\nc\nd" and window_width=2, this will
/// produce ["a\nb", "b\nc", "c\nd"]
pub fn pdf(bytes: &[u8], window_width: usize) -> anyhow::Result<Vec<String>> {
    Ok(pdf_extract::extract_text_from_mem(bytes)?
        .lines()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .windows(window_width)
        .step_by(2)
        .map(|s| s.join("\n"))
        .collect())
}

/// This treats every line of text as a separate document, with no
/// need for windowing or sub-fragmentation.
pub fn text(s: &str) -> anyhow::Result<Vec<String>> {
    Ok(s.lines().map(|s| s.to_string()).collect())
}

#[derive(serde::Deserialize)]
struct JsonlText {
    text: String,
}

/// This treats every jsonl line as a separate document, with no need
/// for windowing or sub-fragmentation.
pub fn jsonl(s: &str) -> anyhow::Result<Vec<String>> {
    Ok(serde_json::Deserializer::from_str(s)
        .into_iter::<JsonlText>()
        .filter_map(|line| match line {
            Ok(JsonlText { text }) => Some(text),
            Err(s) => {
                eprintln!("Error parsing jsonl line {s}");
                None
            }
        })
        .collect())
}
