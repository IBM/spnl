use crate::builtins::Builtin;
use clap::Parser;

#[derive(Parser, Debug, serde::Serialize)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// File to process
    #[arg(required_unless_present("builtin"))]
    pub file: Option<String>,

    /// Builtin to run
    #[arg(value_enum, short, long)]
    pub builtin: Option<Builtin>,

    /// Generative Model
    #[arg(short, long, default_value = "ollama/granite3.3:2b")]
    pub model: String,

    /// Embedding Model
    #[arg(short, long, default_value = "ollama/mxbai-embed-large:335m")]
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
    #[arg(short = 'k', long, default_value_t = 1)]
    pub chunk_size: usize,

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

    /// The RAG indexing scheme
    #[cfg(feature = "rag")]
    #[arg(value_enum, short, long)]
    pub indexer: Option<spnl::Indexer>,

    /// Re-emit the compiled query
    #[arg(short, long, default_value_t = false)]
    pub show_query: bool,

    /// Report query execution time to stderr
    #[arg(long, default_value_t = false)]
    pub time: bool,

    /// Verbose output
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Dry run (do not execute query)?
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}
