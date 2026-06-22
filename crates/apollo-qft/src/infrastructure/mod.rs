//! Infrastructure layer for quantum Fourier transforms.

/// Kernel primitives.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
