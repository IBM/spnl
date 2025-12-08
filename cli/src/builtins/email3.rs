use rustyline::error::ReadlineError;

use crate::args::Args;
use spnl::{ir::Query, spnl};

// https://github.ibm.com/AI4BA/agentic-policy
pub fn query(args: Args) -> Query {
    let Args {
        model,
        n,
        temperature,
        max_tokens,
        ..
    } = args;

    let outer_max_tokens = if args.time.is_some() { Some(1) } else { None };

    let mut rl = rustyline::DefaultEditor::new().unwrap();
    if rl.load_history("history.txt").is_err() {
        println!("No previous history.");
    }
    let prompt = match rl.readline("Tell me about yourself: ") {
        Ok(line) => {
            rl.add_history_entry(line.as_str()).unwrap();
            line
        }
        Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => ::std::process::exit(0),
        Err(err) => panic!("{}", err),
    };
    rl.append_history("history.txt").unwrap();

    let candidate_emails = spnl!(
        repeat n
             model (seq
                       (system (file "email3-generate-system-prompt.txt"))
                       (user prompt))

              temperature max_tokens
    );

    spnl!(g model (seq
                   (print "Evaluating candidate emails")
                   (system (file "email3-evaluate-system-prompt.txt"))
                   candidate_emails
                   (print "Generate candidate emails in parallel"))

          temperature outer_max_tokens
    )
}
