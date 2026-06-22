#![warn(missing_docs)]
//! Discrete and continuous wavelet transforms for Apollo.
//!
//! Wavelet transforms provide multiresolution analysis: high-frequency content
//! is localized with short effective windows, while low-frequency content is
//! represented over longer support. This crate owns Apollo's wavelet domain
//! metadata, DWT/CWT plans, numerical kernels, and verification.
//!
//! Implemented surfaces:
//! - Orthogonal 1D DWT with periodic boundaries for Haar and Daubechies-4.
//! - Multilevel inverse DWT for perfect reconstruction on power-of-two lengths.
//! - Real-valued 1D CWT using Ricker and Morlet analysis wavelets.

/// Application-layer wavelet plans.
pub mod application;
/// Domain contracts and metadata.
pub mod domain;
/// Infrastructure kernels.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::cwt::CwtPlan;
pub use application::execution::plan::dwt::{DwtLetoCoefficients, DwtPlan, WaveletStorage};
pub use domain::contracts::error::{WaveletError, WaveletResult};
pub use domain::metadata::wavelet::{ContinuousWavelet, DiscreteWavelet};
pub use domain::spectrum::coefficients::{CwtCoefficients, DwtCoefficients};

/// GPU-accelerated backend using WGPU.
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    pub use crate::infrastructure::transport::gpu::*;
}
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::{*};
