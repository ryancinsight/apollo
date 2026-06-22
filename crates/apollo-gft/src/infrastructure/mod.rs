//! Infrastructure layer for graph Fourier transforms.

/// Kernel primitives.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
