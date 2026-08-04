#![warn(missing_docs)]
//! WGPU backend boundary for Apollo GFT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the GFT kernel, its domain names, and
//! the basis-parameterized execution surface. Every GFT operation takes
//! the graph Fourier basis `U` as an operand, so the marker implements
//! only the planner contract and the surface lives on the
//! [`BasisTransform`] extension trait.

mod execution;
/// Infrastructure boundary for the GFT kernel.
pub mod infrastructure;
mod surface;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::GftGpuKernel;
pub use surface::BasisTransform;

/// Metadata-preserving WGPU plan descriptor.
pub type GftWgpuPlan = apollo_fft::WgpuTransformPlan<GftGpuKernel>;

/// WGPU backend descriptor.
pub type GftWgpuBackend = apollo_fft::WgpuTransformBackend<GftGpuKernel>;
