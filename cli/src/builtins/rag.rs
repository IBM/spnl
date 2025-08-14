pub fn query(args: crate::args::Args) -> anyhow::Result<spnl::Query> {
    let crate::args::Args {
        model,
        embedding_model,
        ..
    } = args;

    // The question to augment. We use a default value that pertains
    // to the Prompt Declaration Language (PDL)
    // documentation. https://github.com/IBM/prompt-declaration-language
    let prompt = args
        .prompt
        .unwrap_or_else(|| "Does PDL have a contribute keyword?".into());

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
Respond with either "UNANSWERABLE" or "ANSWERABLE" depending on
whether or not the given documents are sufficient to answer the
question and cite the Relevant Documents used to answer the
question. Do not rely on your pre-trained knowledge.
"#;

    Ok(spnl::spnl!(
        g model
            (cross
             (system system_prompt)
             (with embedding_model (user prompt) docs))
    ))
}
