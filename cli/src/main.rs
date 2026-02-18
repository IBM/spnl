use clap::Parser;

use crate::args::{Args, Commands, FullArgs};
use crate::builtins::*;
use spnl::{ExecuteOptions, SpnlError, execute, ir::from_str, ir::pretty_print, optimizer::hlo};

#[cfg(feature = "gce")]
use crate::args::ImageCommands;
#[cfg(feature = "vllm")]
use crate::args::VllmCommands;
#[cfg(any(feature = "k8s", feature = "gce"))]
use crate::args::VllmTarget;
#[cfg(feature = "vllm")]
use spnl::vllm;
#[cfg(feature = "gce")]
use spnl::vllm::gce as gce_vllm;
#[cfg(feature = "k8s")]
use spnl::vllm::k8s as k8s_vllm;

#[cfg(feature = "rag")]
use spnl::AugmentOptionsBuilder;

mod args;
#[cfg(feature = "bench")]
mod bench;
mod builtins;

#[tokio::main]
async fn main() -> Result<(), SpnlError> {
    env_logger::init();
    dotenv::dotenv().ok();

    let args = FullArgs::parse();

    match args.command {
        Commands::Run(run_args) => run(run_args).await,

        #[cfg(feature = "bench")]
        Commands::Bench { command } => bench::run(command).await,

        #[cfg(feature = "local")]
        Commands::List => list_local_models(),

        #[cfg(feature = "vllm")]
        Commands::Vllm { command } => match command.as_ref() {
            #[cfg(any(feature = "k8s", feature = "gce"))]
            VllmCommands::Up {
                target,
                name,
                model,
                #[cfg(feature = "k8s")]
                tokenizer,
                hf_token,
                gpus: _gpus,
                local_port: _local_port,
                remote_port: _remote_port,
                #[cfg(feature = "gce")]
                gce_config,
                ..
            } => match target {
                #[cfg(feature = "k8s")]
                VllmTarget::K8s => {
                    k8s_vllm::up(
                        k8s_vllm::UpArgsBuilder::default()
                            .name(name.name.clone())
                            .namespace(name.namespace.clone())
                            .model(model.clone())
                            .tokenizer(tokenizer.clone())
                            .hf_token(hf_token.clone())
                            .gpus(*_gpus)
                            .local_port(*_local_port)
                            .remote_port(*_remote_port)
                            .build()?,
                    )
                    .await
                }
                #[cfg(feature = "gce")]
                VllmTarget::Gce => {
                    gce_vllm::up(
                        gce_vllm::UpArgsBuilder::default()
                            .name(name.name.clone())
                            .model(model.clone())
                            .hf_token(hf_token.clone())
                            .local_port(*_local_port)
                            .config(gce_config.clone())
                            .build()?,
                    )
                    .await
                }
            },
            #[cfg(any(feature = "k8s", feature = "gce"))]
            VllmCommands::Down {
                target,
                name,
                #[cfg(feature = "gce")]
                gce_config,
            } => match target {
                #[cfg(feature = "k8s")]
                VllmTarget::K8s => k8s_vllm::down(&name.name, name.namespace.clone()).await,
                #[cfg(feature = "gce")]
                VllmTarget::Gce => {
                    gce_vllm::down(&name.name, name.namespace.clone(), gce_config.clone()).await
                }
            },
            #[cfg(feature = "gce")]
            VllmCommands::Image { command } => match command.as_ref() {
                ImageCommands::Create {
                    target,
                    force,
                    image_name,
                    image_family,
                    llmd_version,
                    gce_config,
                } => match target {
                    VllmTarget::Gce => {
                        let image_name = gce_vllm::create_image(
                            gce_vllm::ImageCreateArgsBuilder::default()
                                .force_overwrite(*force)
                                .image_name(image_name.clone())
                                .image_family(image_family.clone())
                                .llmd_version(llmd_version.clone())
                                .vllm_org(gce_config.vllm_org.clone())
                                .vllm_repo(gce_config.vllm_repo.clone())
                                .vllm_sha(gce_config.vllm_sha.clone())
                                .config(gce_config.clone())
                                .build()?,
                        )
                        .await?;
                        println!("Image created successfully: {}", image_name);
                        Ok(())
                    }
                    #[cfg(feature = "k8s")]
                    VllmTarget::K8s => Err(anyhow::anyhow!(
                        "Image creation is only supported for GCE target"
                    )),
                },
            },
            VllmCommands::Patchfile => vllm::patchfile().await,
        },
    }
}

async fn run(args: Args) -> Result<(), SpnlError> {
    let verbose = args.verbose;
    let show_only = args.show_query;
    let dry_run = args.dry_run;
    let rp = ExecuteOptions {
        time: args.time,
        prepare: Some(args.prepare),
        silent: false,
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

    let is_timing = args.time;

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

    execute(&query, &rp).await.map(|res| {
        if !is_timing && !res.to_string().is_empty() {
            println!("{res}");
        }
        Ok(())
    })?
}

#[cfg(feature = "local")]
fn list_local_models() -> Result<(), SpnlError> {
    use tabled::{Table, Tabled, settings::Style};

    #[derive(Tabled)]
    struct ModelRow {
        #[tabled(rename = "NAME")]
        name: String,
        #[tabled(rename = "CACHED")]
        cached: String,
        #[tabled(rename = "ID")]
        id: String,
    }

    let models = spnl::generate::backend::prettynames::list_all_models();

    let rows: Vec<ModelRow> = models
        .iter()
        .map(|(pretty_name, hf_name, is_cached)| ModelRow {
            name: pretty_name.to_string(),
            cached: if *is_cached { "✓" } else { "-" }.to_string(),
            id: hf_name.to_string(),
        })
        .collect();

    if rows.is_empty() {
        println!("No local models available");
        return Ok(());
    }

    let table = Table::new(rows).with(Style::blank()).to_string();

    // Apply colors: green checkmarks for cached, dim gray dash for not cached
    let colored_table = table
        .replace("✓", "\x1b[32m✓\x1b[0m")
        .replace("-", "\x1b[2m-\x1b[0m");

    println!("{}", colored_table);
    Ok(())
}
