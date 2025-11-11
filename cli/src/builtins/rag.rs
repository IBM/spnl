pub fn query(args: crate::args::Args) -> anyhow::Result<spnl::Query> {
    let crate::args::Args {
        model,
        embedding_model,
        temperature,
        max_tokens,
        ..
    } = args;

    // The question to augment. We use a default value that pertains
    // to the Prompt Declaration Language (PDL)
    // documentation. https://github.com/IBM/prompt-declaration-language
    let prompt = format!(
        "Question: {}",
        args.prompt
            .unwrap_or_else(|| "Does PDL have a contribute keyword?".into())
    );

    // The corpus to mine for augmentations.
    let docs = if let Some(docs) = args.document {
        docs.into_iter()
            .map(::std::path::absolute)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        // Default value, which is the PDL documentation (see link above)
        vec![::std::path::PathBuf::from("./rag-doc1.pdf".to_string())]
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
            temperature max_tokens
    ))
}
