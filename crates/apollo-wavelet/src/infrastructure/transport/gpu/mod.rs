#![warn(missing_docs)]
//! WGPU backend boundary for Apollo Wavelet.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns only the Haar kernels and their domain
//! names.

/// Infrastructure boundary for the Haar kernels.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::{HaarDwtPlan, WaveletGpuKernel};

/// Metadata-preserving WGPU plan descriptor.
pub type WaveletWgpuPlan = apollo_fft::WgpuTransformPlan<WaveletGpuKernel>;

/// WGPU backend descriptor.
pub type WaveletWgpuBackend = apollo_fft::WgpuTransformBackend<WaveletGpuKernel>;
