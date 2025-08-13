pub fn query(args: crate::args::Args) -> anyhow::Result<spnl::Query> {
    let crate::args::Args {
        n,
        reverse,
        temperature,
        ..
    } = args;

    let model = "spnl/ldsjmdy/Tulu3-Block-FT";
    let max_tokens = 1;

    let frag1 = "Sequence Transduction Models";
    let frag2 = "Template-Assisted Selective Epitaxy";

    let num_repeats = (n / 10).try_into()?;

    let fraga = if reverse { frag2 } else { frag1 };
    let fragb = if reverse { frag1 } else { frag2 };
    let doca = ::std::iter::repeat_n(fraga, num_repeats)
        .collect::<Vec<_>>()
        .join(" ");
    let docb = ::std::iter::repeat_n(fragb, num_repeats)
        .collect::<Vec<_>>()
        .join(" ");

    let system_prompt = r#"
You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question."#;

    let question = r#"
Please write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant).
Question: Tell me which one concerns deep learning. Indicate your answer with a number in brackets."#;

    Ok(spnl::spnl!(g model
     (cross
      (system system_prompt)
      (plus (user doca) (user docb))
      (user question)
     )
     temperature max_tokens
    ))
}
