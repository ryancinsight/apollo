#![warn(missing_docs)]
//! Spherical harmonic transform plans for Apollo.
//!
//! `apollo-sht` owns spherical-surface sampling metadata, complex coefficient
//! storage, Gauss-Legendre quadrature, orthonormal spherical harmonic kernels,
//! and forward/inverse transform plans.

/// Application-layer SHT plans.
pub mod application;
/// Domain contracts and metadata.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::sht::{ShtComplexStorage, ShtPlan, ShtRealStorage};
pub use domain::contracts::error::{ShtError, ShtResult};
pub use domain::metadata::grid::SphericalGridSpec;
pub use domain::spectrum::coefficients::SphericalHarmonicCoefficients;

/// GPU-accelerated backend using WGPU.
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    pub use crate::infrastructure::transport::gpu::*;
}
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::{*};
