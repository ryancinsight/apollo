//! Infrastructure layer for spherical harmonic transforms.

/// Kernel namespace.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
