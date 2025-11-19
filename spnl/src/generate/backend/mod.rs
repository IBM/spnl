#[cfg(feature = "openai")]
pub(crate) mod openai;

#[cfg(feature = "spnl-api")]
pub(crate) mod spnl;

pub(crate) mod capabilities;

mod progress;
