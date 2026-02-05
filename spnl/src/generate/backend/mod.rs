#[cfg(feature = "openai")]
pub(crate) mod openai;

#[cfg(feature = "spnl-api")]
pub(crate) mod spnl;

#[cfg(feature = "candle")]
pub(crate) mod candle;

// Shared components used across backends
#[cfg(feature = "candle")]
pub(crate) mod shared;

pub(crate) mod capabilities;

mod progress;
