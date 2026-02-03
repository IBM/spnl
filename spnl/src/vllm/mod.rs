mod patch;

pub use patch::patchfile;

#[cfg(feature = "gce")]
pub mod gce;

#[cfg(feature = "k8s")]
pub mod k8s;

// Made with Bob
