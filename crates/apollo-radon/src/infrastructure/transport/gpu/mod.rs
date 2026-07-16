#![warn(missing_docs)]
//! WGPU backend boundary for Apollo Radon.
//!
//! This crate owns GPU capability and plan descriptors for this transform domain.
//! It implements both forward parallel-beam Radon projection and adjoint
//! backprojection on the GPU via WGSL compute shaders.
//! Mathematical contracts remain in `apollo-radon`.

/// Application-layer WGPU plan descriptors.
pub mod application;
/// Domain contracts for WGPU execution.
pub mod domain;
/// Infrastructure boundary for WGPU device acquisition.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::plan::RadonWgpuPlan;
pub use domain::capabilities::WgpuCapabilities;
pub use domain::error::{WgpuError, WgpuResult};
pub use infrastructure::device::RadonWgpuBackend;

/// CPU transform marker proving dependency direction into the owning transform crate.
pub type CpuTransformMarker = crate::RadonPlan;
