//! NUFFT infrastructure layer.

/// Concrete kernel primitives.
pub mod kernel;

/// Transport-level backend adapters.
#[cfg(feature = "wgpu")]
pub mod transport;
