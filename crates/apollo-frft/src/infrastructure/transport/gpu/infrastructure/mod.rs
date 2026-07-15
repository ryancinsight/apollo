//! WGPU infrastructure.

/// Hephaestus WGPU device acquisition and host boundary.
pub mod device;
/// Typed Hephaestus kernel for the direct FrFT transform.
pub mod kernel;
/// Typed Hephaestus kernel for the unitary FrFT transform.
pub mod unitary_kernel;
