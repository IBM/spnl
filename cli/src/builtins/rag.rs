use itertools::Itertools;

pub fn query(args: crate::args::Args) -> anyhow::Result<spnl::ir::Query> {
    let crate::args::Args {
        model,
        embedding_model,
        temperature,
        max_tokens,
        chunk_size,
        ..
    } = args;

    // TODO use args.max_tokens to govern the length of the documents
    let outer_max_tokens = if args.time.is_some() {
        Some(1)
    } else {
        Some(max_tokens)
    };

    // The question to augment. We use a default value that pertains
    // to the Prompt Declaration Language (PDL)
    // documentation. https://github.com/IBM/prompt-declaration-language
    let prompt = format!(
        "Question: {}",
        args.prompt.unwrap_or_else(|| {
            "can i use my flexible savings account to pay for health insurance premiums?"
                .to_string()
        })
    );

    // The corpus to mine for augmentations.
    let docs = if let Some(docs) = args.document {
        docs.into_iter()
            .map(::std::path::absolute)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|filepath| spnl::spnl!(fetchn filepath))
            .collect::<Vec<_>>()
    } else {
        // Default value
        vec![(
            format!(
                "fiqa-first100lines-chunksize-{}.txt",
                chunk_size
                    .map(|s| format!("{s}"))
                    .unwrap_or("none".to_string())
            ),
            spnl::ir::Document::Text(
                spnl::windowing::jsonl(include_str!("fiqa-first100lines.jsonl"))?
                    .into_iter()
                    .map(|line| {
                        if chunk_size.is_none() || line.len() == chunk_size.unwrap() {
                            line.to_string()
                        } else {
                            let width = chunk_size.unwrap_or(1000); // this could probably safely be `.unwrap()`
                            if line.len() < width {
                                format!(
                                    "{}{}",
                                    line.repeat(width / line.len()),
                                    &line[0..width % line.len()]
                                )
                            } else {
                                line[0..width].to_string()
                            }
                        }
                    })
                    .join("\n"),
            ),
        )]
    };

    let system_prompt = r#"
Your answer questions using information from the given Relevant Documents, and cite them. For example:

Question: How do trees grow?
Answer: Via carbon dioxide.
Citations: @base-foo-37, @raptor-bar-52

Question: How does hair grow?
Answer: Slowly.
Citations: @base-baz-2, @raptor-glam-8
"#;

    Ok(spnl::spnl!(
        g model
            (cross
             (system system_prompt)
             (with embedding_model (user prompt) docs))
            temperature outer_max_tokens
    ))
}
