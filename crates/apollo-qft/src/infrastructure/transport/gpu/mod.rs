#![warn(missing_docs)]
//! WGPU backend boundary for Apollo QFT.
//!
//! This crate owns GPU capability and plan descriptors for this transform domain.
//! Mathematical contracts remain in `apollo-qft`.

/// Application-layer WGPU plan descriptors.
pub mod application;
/// Domain contracts for WGPU execution.
pub mod domain;
/// Infrastructure boundary for WGPU device acquisition.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use application::plan::QftWgpuPlan;
pub use domain::capabilities::WgpuCapabilities;
pub use domain::error::{WgpuError, WgpuResult};
pub use infrastructure::device::QftWgpuBackend;
