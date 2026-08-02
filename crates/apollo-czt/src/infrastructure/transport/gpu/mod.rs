#![warn(missing_docs)]
//! WGPU backend boundary for Apollo CZT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns only the chirp-z kernels and their
//! domain names. `Sample` and `Bin` are both [`apollo_fft::Complex32`],
//! and the plan payload carries the spiral parameters, so input and
//! output lengths differ freely per the chirp-z contract.

/// Infrastructure boundary for the chirp-z kernels.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::{ChirpPlan, CztGpuKernel};

/// Metadata-preserving WGPU plan descriptor.
pub type CztWgpuPlan = apollo_fft::WgpuTransformPlan<CztGpuKernel>;

/// WGPU backend descriptor.
pub type CztWgpuBackend = apollo_fft::WgpuTransformBackend<CztGpuKernel>;
