#![warn(missing_docs)]
//! WGPU backend boundary for Apollo STFT.
//!
//! Provides GPU-accelerated forward and inverse STFT execution on f32 signals.
//! CPU reference implementation and domain contracts live in `apollo-stft`.

/// Application-layer WGPU plan descriptors.
pub mod application;
/// Domain contracts for WGPU execution.
pub mod domain;
/// Infrastructure boundary for WGPU device acquisition and kernel execution.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::plan::StftWgpuPlan;
pub use domain::capabilities::WgpuCapabilities;
pub use domain::error::{WgpuError, WgpuResult};
pub use eunomia::Complex32;
pub use infrastructure::buffers::StftGpuBuffers;
pub use infrastructure::device::{wgpu_available, StftWgpuBackend};
pub use infrastructure::kernel::StftGpuKernel;

/// CPU transform marker proving dependency direction into the owning transform crate.
pub type CpuTransformMarker = crate::StftPlan;
