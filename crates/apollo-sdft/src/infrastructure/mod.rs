//! Infrastructure layer for sliding DFT transforms.

/// Kernel namespace.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
