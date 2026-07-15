#![warn(missing_docs)]
//! Non-uniform FFT utilities and reusable NUFFT plans.
//!
//! This crate owns NUFFT domain descriptors, Kaiser-Bessel infrastructure, and
//! application-level Type-1/Type-2 NUFFT plans. Dense FFT execution is delegated
//! to `apollo-fft`; NUFFT-specific logic does not live in `apollo-fft`.

/// Application-layer NUFFT plans and execution surfaces.
pub mod application;
/// NUFFT domain descriptors and error contracts.
pub mod domain;
/// Concrete NUFFT kernel primitives.
pub mod infrastructure;
#[cfg(test)]
mod verification;

pub use application::execution::transform::dimension_1d::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type2_1d, nufft_type2_1d_fast, NufftComplexStorage,
    NufftPlan1D,
};
pub use application::execution::transform::dimension_3d::{
    nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_3d, nufft_type2_3d_fast, NufftPlan3D,
};
pub use domain::metadata::grid::{UniformDomain1D, UniformGrid3D};

/// Default kernel half-width.
pub const DEFAULT_NUFFT_KERNEL_WIDTH: usize = 6;

/// Default oversampling factor.
pub const DEFAULT_NUFFT_OVERSAMPLING: usize = 2;

#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
