pub mod ir;

#[cfg(feature = "ffi")]
pub mod ffi;
#[cfg(feature = "ffi")]
pub use ffi::*;

#[cfg(feature = "run")]
mod execute;
#[cfg(feature = "run")]
pub use execute::*;

// TODO generate feature?
#[cfg(feature = "run")]
mod generate;
#[cfg(feature = "run")]
pub use generate::WhatToTime;

// TODO optimizer feature?
pub mod optimizer;

#[cfg(feature = "rag")]
mod augment;
#[cfg(feature = "rag")]
pub use augment::{AugmentOptionsBuilder, Indexer};
