//! Infrastructure layer for wavelet transforms.

/// Numerical kernels.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
