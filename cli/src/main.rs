use clap::Parser;

use crate::args::Args;
use crate::builtins::*;
use spnl::{
    ExecuteOptions, SpnlError, WhatToTime, execute, ir::from_str, ir::pretty_print, optimizer::hlo,
};

#[cfg(feature = "rag")]
use spnl::AugmentOptionsBuilder;

mod args;
mod builtins;

#[tokio::main]
async fn main() -> Result<(), SpnlError> {
    let args = Args::parse();
    let verbose = args.verbose;
    let show_only = args.show_query;
    let dry_run = args.dry_run;

    let rp = ExecuteOptions {
        time: args.time.clone(),
        prepare: Some(args.prepare),
    };

    let hlo_options = hlo::Options {
        #[cfg(feature = "rag")]
        aug: AugmentOptionsBuilder::default()
            .indexer(args.indexer.clone().unwrap_or_default())
            .verbose(args.verbose)
            .max_aug(args.max_aug)
            .shuffle(args.shuffle)
            .vecdb_uri(args.vecdb_uri.clone())
            .vecdb_table(
                args.builtin
                    .clone()
                    .map(|builtin| format!("builtin.{builtin:?}"))
                    .unwrap_or_else(|| args.file.clone().unwrap_or("default".to_string())),
            )
            .build()?,
    };

    let is_timing = args.time.is_some();
    let start_time = if let Some(WhatToTime::All) = args.time {
        Some(::std::time::Instant::now())
    } else {
        None
    };

    let query = hlo::optimize(
        &match args.builtin {
            Some(Builtin::BulkMap) => bulk_map::query(args),
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
        &hlo_options,
    )
    .await?;

    if show_only {
        pretty_print(&query)?;
    } else if verbose {
        ptree::write_tree(&query, ::std::io::stderr())?;
    }

    if show_only || dry_run {
        return Ok(());
    }

    let res = execute(&query, &rp).await.map(|res| {
        if !is_timing && !res.to_string().is_empty() {
            println!("{res}");
        }
        Ok(())
    })?;

    if let Some(start_time) = start_time {
        println!("AllTime {} ns", start_time.elapsed().as_nanos());
    }

    res
}
