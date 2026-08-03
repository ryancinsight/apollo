#![warn(missing_docs)]
//! WGPU backend boundary for Apollo Radon.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the Radon kernels, their domain names,
//! and the projection surface. Every operation takes the projection
//! angle array as an operand, so the marker implements only the planner
//! contract and the surface lives on [`ProjectionExecution`].

mod execution;
mod geometry;
/// Infrastructure boundary for the Radon kernels.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use execution::ProjectionExecution;
pub use geometry::GeometryPlan;
pub use infrastructure::kernel::RadonGpuKernel;

/// Metadata-preserving WGPU plan descriptor.
pub type RadonWgpuPlan = apollo_fft::WgpuTransformPlan<RadonGpuKernel>;

/// WGPU backend descriptor.
pub type RadonWgpuBackend = apollo_fft::WgpuTransformBackend<RadonGpuKernel>;
