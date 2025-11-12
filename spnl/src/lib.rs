// TODO ir feature?
pub mod ir;

#[cfg(feature = "run")]
mod execute;
#[cfg(feature = "run")]
pub use execute::*;

// TODO generate feature?
#[cfg(feature = "run")]
mod generate;

// TODO optimizer feature?
pub mod optimizer;

#[cfg(feature = "rag")]
mod augment;
#[cfg(feature = "rag")]
pub use augment::{AugmentOptionsBuilder, Indexer};

#[cfg(feature = "pypi")]
mod python;
#[cfg(feature = "pypi")]
pub use python::spnl_py;
