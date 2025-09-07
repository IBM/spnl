mod query;
pub use query::*;

#[cfg(feature = "run")]
mod execute;
#[cfg(feature = "run")]
pub use execute::*;

// TODO generate feature?
#[cfg(feature = "run")]
mod generate;

// TODO plan feature?
#[cfg(feature = "run")]
mod plan;

// TODO plan feature?
#[cfg(feature = "run")]
pub use plan::*;

#[cfg(feature = "rag")]
mod augment;
#[cfg(feature = "rag")]
pub use augment::{AugmentOptionsBuilder, Indexer};

#[cfg(feature = "pull")]
pub mod pull;

#[cfg(feature = "tok")]
pub mod tokenize;

#[cfg(feature = "tok")]
pub mod chat_template;

#[cfg(feature = "pypi")]
mod python;
#[cfg(feature = "pypi")]
pub use python::spnl_py;

#[cfg(feature = "lisp")]
mod lisp;
