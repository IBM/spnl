use crate::args::Args;
use spnl::{Query, spnl};

pub fn query(args: Args) -> Query {
    let Args {
        model, temperature, ..
    } = args;

    spnl!(gx model (cross (system "You are a helpful chat bot") (ask "❯ ")) temperature)
}
