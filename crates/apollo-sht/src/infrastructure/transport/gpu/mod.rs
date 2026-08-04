#![warn(missing_docs)]
//! WGPU backend boundary for Apollo SHT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the SHT kernels, their domain names, and
//! the harmonic surface. Forward analysis produces
//! [`SphericalHarmonicCoefficients`] (a `Complex64` domain structure) and
//! inverse synthesis consumes it with exact-representability checks, so
//! the surface extends the scaffold through [`HarmonicExecution`] rather
//! than instantiating its slice contract.

mod execution;
/// Infrastructure boundary for the SHT kernels.
pub mod infrastructure;
mod spherical;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use execution::HarmonicExecution;
pub use infrastructure::kernel::ShtGpuKernel;
pub use spherical::SphericalPlan;

/// Metadata-preserving WGPU plan descriptor.
pub type ShtWgpuPlan = apollo_fft::WgpuTransformPlan<ShtGpuKernel>;

/// WGPU backend descriptor.
pub type ShtWgpuBackend = apollo_fft::WgpuTransformBackend<ShtGpuKernel>;
