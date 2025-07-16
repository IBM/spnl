pub fn demo(args: crate::args::Args) -> Result<spnl::Query, Box<dyn ::std::error::Error>> {
    let crate::args::Args {
        model,
        embedding_model,
        question,
        document,
        ..
    } = args;

    let docs: Vec<String> = if let Some(docs) = document {
        docs.into_iter()
            .map(::std::path::absolute)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|doc| doc.into_os_string().into_string().expect("string"))
            .collect()
    } else {
        vec!["./rag-doc1.pdf".to_string()]
    };

    Ok(spnl::spnl!(
        g model
            (cross
             (system r#"You answer only with either "UNANSWERABLE" or "ANSWERABLE" depending on whether or not the given documents are sufficient to answer the question."#)
             (with embedding_model (user question) docs))
    ))
}
