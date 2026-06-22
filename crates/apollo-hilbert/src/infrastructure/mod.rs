//! Hilbert infrastructure layer.

/// Concrete kernels.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
