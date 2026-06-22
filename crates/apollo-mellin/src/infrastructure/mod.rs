//! Infrastructure layer for Mellin transforms.

/// Kernel namespace.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
