#[cfg(feature = "pypi")]
pub mod python;

#[cfg(feature = "pypi")]
pub use python::spnl_py;
