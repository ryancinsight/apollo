#![warn(missing_docs)]
//! WGPU backend boundary for Apollo DHT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns only the DHT kernel and its domain names.

/// Infrastructure boundary for the DHT kernel.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::DhtGpuKernel;

/// Metadata-preserving WGPU plan descriptor.
pub type DhtWgpuPlan = apollo_fft::WgpuTransformPlan<DhtGpuKernel>;

/// WGPU backend descriptor.
pub type DhtWgpuBackend = apollo_fft::WgpuTransformBackend<DhtGpuKernel>;
