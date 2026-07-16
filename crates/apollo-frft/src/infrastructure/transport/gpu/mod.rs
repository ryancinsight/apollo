#![warn(missing_docs)]
//! WGPU backend boundary for Apollo FrFT.
//!
//! This crate owns GPU capability and plan descriptors for this transform domain.
//! Mathematical contracts remain in `apollo-frft`.

/// Application-layer WGPU plan descriptors.
pub mod application;
/// Domain contracts for WGPU execution.
pub mod domain;
/// Infrastructure boundary for WGPU device acquisition.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use application::plan::FrftWgpuPlan;
pub use application::plan::UnitaryFrftWgpuPlan;
pub use domain::capabilities::WgpuCapabilities;
pub use domain::error::{WgpuError, WgpuResult};
pub use infrastructure::device::FrftWgpuBackend;

/// CPU transform marker proving dependency direction into the owning transform crate.
pub type CpuTransformMarker = crate::FrftPlan;
