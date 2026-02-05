// Shared components used by both Candle and MLX backends

pub mod config;
pub mod download;
pub mod download_progress;
pub mod template_tokenizer;

pub use config::{GenericConfig, detect_architecture};
pub use download::download_model_files;
pub use template_tokenizer::tokenize_with_chat_template;

// Made with Bob
