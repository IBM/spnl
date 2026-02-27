use crate::args::Args;
use spnl::{ir::Query, spnl};

pub fn query(args: Args) -> anyhow::Result<Query> {
    let Args { model, n, .. } = args;

    let chunk_size = args.chunk_size.unwrap_or(1);

    Ok(spnl!(combine model
             (plus (chunk chunk_size
                    (prefix "Question " (take n (file "./gsm8k-questions.json")))
                    (lambda (parts)
                     (extract model (length parts)
                      (g model
                       (seq
                        (system "You are an AI that reasons about math word problems")
                        (plus parts)))))))
    ))
}
