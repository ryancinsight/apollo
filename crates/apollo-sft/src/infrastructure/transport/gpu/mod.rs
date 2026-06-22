#![warn(missing_docs)]
//! WGPU backend boundary for Apollo SFT.
//!
//! This crate owns GPU capability and plan descriptors for this transform domain.
//! Mathematical contracts remain in `apollo-sft`.

/// Application-layer WGPU plan descriptors.
pub mod application;
/// Domain contracts for WGPU execution.
pub mod domain;
/// Infrastructure boundary for WGPU device acquisition.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::plan::SftWgpuPlan;
pub use domain::capabilities::WgpuCapabilities;
pub use domain::error::{WgpuError, WgpuResult};
pub use infrastructure::device::{wgpu_available, SftWgpuBackend};

/// CPU transform marker proving dependency direction into the owning transform crate.
pub type CpuTransformMarker = crate::SparseFftPlan;
