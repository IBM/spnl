use crate::builtins::Builtin;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug, serde::Serialize)]
#[command(version, about, long_about = None)]
pub struct FullArgs {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug, serde::Serialize)]
pub enum Commands {
    /// Run a query
    Run(Args),

    /// Bring up vLLM in a Kubernetes cluster
    #[cfg(feature = "vllm")]
    Vllm {
        #[command(subcommand)]
        command: VllmCommands,
    },
}

#[derive(Parser, Debug, serde::Serialize)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// File to process
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    /// Builtin to run
    #[arg(
        value_enum,
        short,
        long,
        env = "SPNL_BUILTIN",
        required_unless_present("file")
    )]
    pub builtin: Option<Builtin>,

    /// Generative Model
    #[arg(
        short,
        long,
        default_value = "ollama/granite3.3:2b",
        env = "SPNL_MODEL"
    )]
    pub model: String,

    /// Embedding Model
    #[arg(
        short,
        long,
        default_value = "ollama/mxbai-embed-large:335m",
        env = "SPNL_EMBEDDING_MODEL"
    )]
    pub embedding_model: String,

    /// Temperature
    #[arg(short, long, default_value_t = 0.5)]
    pub temperature: f32,

    /// Max Completion/Generated Tokens
    #[arg(short = 'l', long, default_value_t = 100)]
    pub max_tokens: i32,

    /// Number of candidates to consider
    #[arg(short, long, default_value_t = 5)]
    pub n: u32,

    /// Chunk size
    #[arg(short = 'k', long)]
    pub chunk_size: Option<usize>,

    /// Vector DB Url
    #[cfg(feature = "rag")]
    #[arg(long, default_value = "data/spnl")]
    pub vecdb_uri: String,

    /// Reverse order
    #[arg(short, long, default_value_t = false)]
    pub reverse: bool,

    /// Prepare query
    #[arg(long, default_value_t = false)]
    pub prepare: bool,

    /// Question to pose
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Document(s) that will augment the question
    #[cfg(feature = "rag")]
    #[arg(short = 'd', long)]
    pub document: Option<Vec<String>>,

    /// Max augmentations to add to the query
    #[cfg(feature = "rag")]
    #[arg(short = 'x', long, env = "SPNL_RAG_MAX_MATCHES")]
    pub max_aug: Option<usize>,

    /// Randomly shuffle order of fragments
    #[cfg(feature = "rag")]
    #[arg(long, default_value_t = false)]
    pub shuffle: bool,

    /// The RAG indexing scheme
    #[cfg(feature = "rag")]
    #[arg(value_enum, short, long)]
    pub indexer: Option<spnl::Indexer>,

    /// Re-emit the compiled query
    #[arg(short, long, default_value_t = false)]
    pub show_query: bool,

    /// Report query execution time to stderr
    #[arg(value_enum, long)]
    pub time: Option<spnl::WhatToTime>,

    /// Verbose output
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Dry run (do not execute query)?
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}

#[cfg(feature = "vllm")]
#[derive(Subcommand, Debug, serde::Serialize)]
pub enum VllmCommands {
    Up {
        #[command(flatten)]
        name: K8sNameArgs,

        /// Model to serve
        #[arg(short = 'm', long, env = "SPNL_MODEL")]
        model: Option<String>,

        /// HuggingFace token, used to pull model weights
        #[arg(short = 't', long, env = "HF_TOKEN", required = true)]
        hf_token: String,

        /// Number of GPUs to request
        #[arg(long, default_value_t = 1)]
        gpus: u32,

        /// Local port for port forwarding (defaults to 8000)
        #[arg(short = 'p', long, default_value = "8000")]
        local_port: Option<u16>,

        /// Remote port for port forwarding (defaults to 8000)
        #[arg(short = 'r', long, default_value_t = 8000)]
        remote_port: u16,
    },
    Down {
        #[command(flatten)]
        name: K8sNameArgs,
    },
}

#[cfg(feature = "k8s")]
#[derive(clap::Args, Debug, serde::Serialize)]
pub struct K8sNameArgs {
    /// Name of Kubernetes Deployment resource
    #[arg(required = true)]
    pub name: String,

    /// Namespace of Kubernetes Deployment resource
    #[arg(short = 'n', long)]
    pub namespace: Option<String>,
}
