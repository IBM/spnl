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
    #[arg(long, default_value = "data/spnl")]
    pub vecdb_uri: String,

    /// Reverse order
    #[arg(short, long, default_value_t = false)]
    pub reverse: bool,

    /// Prepare query
    #[arg(short, long, default_value_t = false)]
    pub prepare: bool,

    /// Question to pose
    #[arg(short = 'w', long)]
    pub question: Option<String>,

    /// Document that will augment the question
    #[arg(short = 'u', long)]
    pub document: Option<Vec<String>>,

    /// Re-emit the compiled query
    #[arg(short, long, default_value_t = false)]
    pub show_query: bool,

    /// Report query execution time to stderr
    #[arg(long, default_value_t = false)]
    pub time: bool,

    /// Verbose output
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,
}
