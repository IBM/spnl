use crate::args::Args;
use spnl::{Unit, spnl};

pub fn demo(args: Args) -> Unit {
    let Args {
        model, temperature, ..
    } = args;

    spnl!(loop (g model (cross (system "You are a helpful chat bot") (ask "❯ ")) temperature))
}
