#![warn(missing_docs)]
//! WGPU backend boundary for Apollo QFT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns only the QFT kernel and its domain
//! names. The QFT is the first complex-element adopter: `Sample` and
//! `Bin` are both [`apollo_fft::Complex32`], and typed reduced-precision
//! dispatch runs over `[f16; 2]` storage through
//! `apollo_fft::GpuStorage<Complex32>`.

/// Infrastructure boundary for the QFT kernel.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::QftGpuKernel;

/// Metadata-preserving WGPU plan descriptor.
pub type QftWgpuPlan = apollo_fft::WgpuTransformPlan<QftGpuKernel>;

/// WGPU backend descriptor.
pub type QftWgpuBackend = apollo_fft::WgpuTransformBackend<QftGpuKernel>;
