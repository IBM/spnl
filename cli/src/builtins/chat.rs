use crate::args::Args;
use spnl::{ir::Query, spnl};

pub fn query(args: Args) -> Query {
    let Args {
        model,
        temperature,
        max_tokens,
        prompt,
        ..
    } = args;
    let my_prompt = prompt.unwrap_or("what is the sky blue".to_string());

    spnl!(
        g model (cross "Chatting" (system "You are a helpful chat bot") (user my_prompt)) temperature max_tokens
    )
}
