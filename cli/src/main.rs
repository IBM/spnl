use clap::Parser;

use crate::args::Args;
use crate::builtins::*;
use spnl::{
    from_str, pretty_print,
    run::{
        RunParameters,
        plan::{PlanOptions, plan},
        result::SpnlError,
        run,
    },
};

mod args;
mod builtins;

#[tokio::main]
async fn main() -> Result<(), SpnlError> {
    let args = Args::parse();
    let verbose = args.verbose;
    let show_only = args.show_query;

    let rp = RunParameters {
        prepare: Some(args.prepare),
    };

    let plan_options = PlanOptions {
        max_aug: args.max_aug,
        vecdb_uri: args.vecdb_uri.clone(),
        vecdb_table: args
            .builtin
            .clone()
            .map(|builtin| format!("builtin.{builtin:?}"))
            .unwrap_or_else(|| args.file.clone().unwrap_or("default".to_string())),
    };

    let time = if args.time {
        Some(::std::time::Instant::now())
    } else {
        None
    };

    let query = plan(
        &match args.builtin {
            Some(Builtin::Chat) => chat::query(args),
            Some(Builtin::Email) => email::query(args),
            Some(Builtin::Email2) => email2::query(args),
            Some(Builtin::Email3) => email3::query(args),
            Some(Builtin::SWEAgent) => sweagent::query(args).expect("query to be prepared"),
            Some(Builtin::GSM8k) => gsm8k::query(args).expect("query to be prepared"),
            #[cfg(feature = "rag")]
            Some(Builtin::Rag) => rag::query(args).expect("queryto be prepared"),
            #[cfg(feature = "spnl-api")]
            Some(Builtin::Spans) => spans::query(args).expect("query to be prepared"),
            None => {
                use std::io::prelude::*;
                let file = ::std::fs::File::open(args.file.clone().unwrap())?;
                let mut buf_reader = ::std::io::BufReader::new(file);
                let mut contents = String::new();
                buf_reader.read_to_string(&mut contents)?;

                let mut tt = tinytemplate::TinyTemplate::new();
                tt.add_template("file", contents.as_str())?;
                let rendered = tt.render("file", &args)?;
                from_str(rendered.as_str())?
            }
        },
        &plan_options,
    )
    .await?;

    if show_only {
        pretty_print(&query)?;
        return Ok(());
    } else if verbose {
        ptree::write_tree(&query, ::std::io::stderr())?;
    }

    let res = run(&query, &rp).await.map(|res| {
        if !res.to_string().is_empty() {
            println!("{res}");
        }
        Ok(())
    })?;

    if let Some(time) = time {
        eprintln!("{}", time.elapsed().as_millis());
    }

    res
}
