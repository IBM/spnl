#[cfg(feature = "openai")]
pub(crate) mod openai;

#[cfg(feature = "spnl-api")]
pub(crate) mod spnl;

#[cfg(feature = "local")]
pub(crate) mod mistralrs;

#[cfg(feature = "local")]
pub mod prettynames;

pub(crate) mod capabilities;

mod progress;
pub(crate) mod timing;
