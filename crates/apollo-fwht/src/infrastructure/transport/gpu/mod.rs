#![warn(missing_docs)]
//! WGPU backend boundary for Apollo FWHT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns only the FWHT kernel and its domain names.

/// Infrastructure boundary for the FWHT kernel.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::FwhtGpuKernel;

/// Metadata-preserving WGPU plan descriptor.
pub type FwhtWgpuPlan = apollo_fft::WgpuTransformPlan<FwhtGpuKernel>;

/// WGPU backend descriptor.
pub type FwhtWgpuBackend = apollo_fft::WgpuTransformBackend<FwhtGpuKernel>;
