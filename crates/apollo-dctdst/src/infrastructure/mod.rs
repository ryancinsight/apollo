//! Infrastructure layer for DCT/DST transforms.

/// Kernel namespace.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
