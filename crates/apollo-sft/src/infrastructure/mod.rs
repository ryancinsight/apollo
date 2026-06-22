//! Infrastructure kernels used by sparse transform execution.

/// Direct reference kernels.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
