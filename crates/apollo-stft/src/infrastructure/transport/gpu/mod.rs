#![warn(missing_docs)]
//! WGPU backend boundary for Apollo STFT.
//!
//! Provides GPU-accelerated forward and inverse STFT execution on f32 signals.
//! CPU reference implementation and domain contracts live in `apollo-stft`.

/// Application-layer WGPU plan descriptors.
pub mod application;
/// Domain contracts for WGPU execution.
pub mod domain;
/// Infrastructure boundary for provider-built device kernel execution.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use application::plan::StftWgpuPlan;
pub use domain::capabilities::WgpuCapabilities;
pub use domain::error::{WgpuError, WgpuResult};
pub use eunomia::Complex32;
pub use infrastructure::buffers::StftGpuBuffers;
pub use infrastructure::device::StftWgpuBackend;
pub use infrastructure::kernel::StftGpuKernel;
