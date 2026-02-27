// Re-export core types for convenience — excludes `optimizer` and
// `capabilities` to avoid collisions with spnl-run's own modules.
pub use spnl_core::capabilities;
pub use spnl_core::ir;

#[cfg(feature = "run")]
mod execute;
#[cfg(feature = "run")]
pub use execute::*;

// TODO generate feature?
#[cfg(feature = "run")]
pub mod generate;

#[cfg(feature = "run")]
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
