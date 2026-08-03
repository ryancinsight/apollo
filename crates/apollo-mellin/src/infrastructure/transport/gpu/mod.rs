#![warn(missing_docs)]
//! WGPU backend boundary for Apollo Mellin.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the Mellin kernels, their domain names,
//! and the resampled execution surface. Every Mellin operation carries
//! per-call value-domain bounds (the signal or output min/max over which
//! log/exponential resampling runs), so the marker implements only the
//! planner contract and the surface lives on [`ResampledExecution`].

mod execution;
/// Infrastructure boundary for the Mellin kernels.
pub mod infrastructure;
mod scale;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use execution::ResampledExecution;
pub use infrastructure::kernel::MellinGpuKernel;
pub use scale::ScalePlan;

/// Metadata-preserving WGPU plan descriptor.
pub type MellinWgpuPlan = apollo_fft::WgpuTransformPlan<MellinGpuKernel>;

/// WGPU backend descriptor.
pub type MellinWgpuBackend = apollo_fft::WgpuTransformBackend<MellinGpuKernel>;
