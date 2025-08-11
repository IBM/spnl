pub fn demo(args: crate::args::Args) -> Result<spnl::Query, Box<dyn ::std::error::Error>> {
    let crate::args::Args {
        model,
        embedding_model,
        ..
    } = args;

    // The question to augment. We use a default value that pertains
    // to the Prompt Declaration Language (PDL)
    // documentation. https://github.com/IBM/prompt-declaration-language
    let question = args
        .question
        .unwrap_or_else(|| "Does PDL have a contribute keyword?".into());

    // The corpus to mine for augmentations.
    let docs: Vec<String> = if let Some(docs) = args.document {
        docs.into_iter()
            .map(::std::path::absolute)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|doc| doc.into_os_string().into_string().expect("string"))
            .collect()
    } else {
        // Default value, which is the PDL documentation (see link above)
        vec!["./rag-doc1.pdf".to_string()]
    };

    let system_prompt = r#"
Respond with either "UNANSWERABLE" or "ANSWERABLE" depending
on whether or not the given documents are sufficient to answer the
question. Include citation to documents used to service the question.
"#;

    Ok(spnl::spnl!(
        g model
            (cross
             (system system_prompt)
             (with embedding_model (user question) docs))
    ))
}
