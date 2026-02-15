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

    /// List available local models with pretty names (requires 'local' feature)
    #[cfg(feature = "local")]
    List,

    /// Bring up vLLM in a Kubernetes cluster
    #[cfg(feature = "vllm")]
    Vllm {
        #[command(subcommand)]
        command: Box<VllmCommands>,
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
    #[arg(short, long, default_value = "llama3.2:1b", env = "SPNL_MODEL")]
    pub model: String,

    /// Embedding Model
    #[arg(
        short,
        long,
        default_value = "local/google/embeddinggemma-300m",
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

    /// Report timing metrics (TTFT and ITL) to stdout
    #[arg(long)]
    pub time: bool,

    /// Verbose output
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Dry run (do not execute query)?
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}

#[cfg(any(feature = "k8s", feature = "gce"))]
#[derive(clap::ValueEnum, Clone, Debug, serde::Serialize)]
pub enum VllmTarget {
    #[cfg(feature = "k8s")]
    K8s,
    #[cfg(feature = "gce")]
    Gce,
}

#[cfg(feature = "vllm")]
#[derive(Subcommand, Debug, serde::Serialize)]
pub enum VllmCommands {
    #[cfg(any(feature = "k8s", feature = "gce"))]
    Up {
        /// Target platform (k8s or gce)
        #[arg(long, default_value = "k8s")]
        target: VllmTarget,

        #[command(flatten)]
        name: NameArgs,

        /// Model to serve
        #[arg(short = 'm', long, env = "SPNL_MODEL")]
        model: Option<String>,

        /// Tokenizer to use (optional, defaults to model's tokenizer)
        #[arg(long)]
        tokenizer: Option<String>,

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

        /// GCE configuration (only used when target is gce)
        #[cfg(feature = "gce")]
        #[command(flatten)]
        gce_config: spnl::vllm::gce::GceConfig,
    },
    #[cfg(any(feature = "k8s", feature = "gce"))]
    Down {
        /// Target platform (k8s or gce)
        #[arg(long, default_value = "k8s")]
        target: VllmTarget,

        #[command(flatten)]
        name: NameArgs,

        /// GCE configuration (only used when target is gce)
        #[cfg(feature = "gce")]
        #[command(flatten)]
        gce_config: spnl::vllm::gce::GceConfig,
    },
    /// Manage custom images with vLLM pre-installed
    #[cfg(feature = "gce")]
    Image {
        #[command(subcommand)]
        command: Box<ImageCommands>,
    },
    /// Emit vLLM patchfile to stdout
    Patchfile,
}

#[cfg(feature = "gce")]
#[derive(Subcommand, Debug, serde::Serialize)]
pub enum ImageCommands {
    /// Create a custom image with vLLM pre-installed
    Create {
        /// Target platform (only gce is supported)
        #[arg(long, default_value = "gce")]
        target: VllmTarget,

        /// Force overwrite of existing image with the same name
        #[arg(short = 'f', long)]
        force: bool,

        /// Custom image name (defaults to auto-generated from hash)
        #[arg(long)]
        image_name: Option<String>,

        /// Image family
        #[arg(long, default_value = "vllm-spnl")]
        image_family: String,

        /// LLM-D version for patch file
        #[arg(long, default_value = "0.5.0")]
        llmd_version: String,

        /// GCE configuration
        #[command(flatten)]
        gce_config: spnl::vllm::gce::GceConfig,
    },
}

#[cfg(any(feature = "k8s", feature = "gce"))]
#[derive(clap::Args, Debug, serde::Serialize)]
pub struct NameArgs {
    /// Name of the deployment/instance
    #[arg(required = true)]
    pub name: String,

    /// Namespace (k8s only)
    #[arg(short = 'n', long)]
    pub namespace: Option<String>,
}
