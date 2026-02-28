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
pub mod generate;

// TODO optimizer feature?
pub mod optimizer;

#[cfg(feature = "rag")]
mod augment;
#[cfg(feature = "rag")]
pub use augment::{AugmentOptionsBuilder, Indexer};

#[cfg(feature = "k8s")]
pub mod k8s;

#[cfg(feature = "gce")]
pub mod gce;

#[cfg(feature = "vllm")]
pub mod vllm;

/// Model pool management. Only available with the `local` feature.
#[cfg(feature = "local")]
pub mod model_pool {
    /// Unload all models from the global pool, releasing GPU memory.
    pub async fn unload_all() {
        crate::generate::backend::mistralrs::unload_all_models().await
    }
}

/// PIC cache hit/miss statistics (delegates to mistralrs-core).
/// Only available with the `local` feature.
#[cfg(feature = "local")]
pub mod pic_stats {
    /// Read and reset global PIC cache hit/miss counters. Returns `(hits, misses)`.
    pub fn take_cache_stats() -> (u64, u64) {
        mistralrs::pic::take_cache_stats()
    }
}
