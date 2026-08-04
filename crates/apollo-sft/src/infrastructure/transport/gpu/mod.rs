#![warn(missing_docs)]
//! WGPU backend boundary for Apollo SFT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the SFT kernel, its domain names, and
//! the sparse-domain surface. The dense DFT/IDFT instantiate the
//! scaffold directly; sparse selection, sparse reconstruction, and the
//! `Complex64` quantization boundary extend it through
//! [`SparseExecution`], since a [`SparseSpectrum`] is a domain structure,
//! not a slice.

mod execution;
/// Infrastructure boundary for the SFT kernel.
pub mod infrastructure;
mod sparsity;
mod spectrum;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use execution::SparseExecution;
pub use infrastructure::kernel::SftGpuKernel;
pub use sparsity::SparsityPlan;

/// Metadata-preserving WGPU plan descriptor.
pub type SftWgpuPlan = apollo_fft::WgpuTransformPlan<SftGpuKernel>;

/// WGPU backend descriptor.
pub type SftWgpuBackend = apollo_fft::WgpuTransformBackend<SftGpuKernel>;
