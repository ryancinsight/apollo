#![warn(missing_docs)]
//! Mellin transform plans for Apollo.
//!
//! The Mellin transform maps a positive scale-domain signal `f(r)` to
//! moments `M(s) = int_a^b f(r) r^(s-1) dr`. Under the logarithmic change of
//! variables `r = exp(u)`, the imaginary-axis Mellin transform becomes a
//! Fourier transform over `u`, which is the source of scale-shift behavior used
//! in scale-invariant matching.
//!
//! This crate owns positive scale-domain contracts, log-resampling, real
//! Mellin moments, log-frequency Mellin spectra, and value-semantic tests.

/// Application-layer Mellin plans.
pub mod application;
/// Domain contracts and metadata.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::mellin::{MellinPlan, MellinSpectrum, MellinStorage};
pub use domain::contracts::error::{MellinError, MellinResult};
pub use domain::metadata::scale::MellinScaleConfig;
pub use infrastructure::kernel::resample::{
    calculate_log_resample, exp_resample, inverse_log_frequency_spectrum, log_frequency_spectrum,
    mellin_moment,
};

/// GPU-accelerated backend using WGPU.
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    pub use crate::infrastructure::transport::gpu::*;
}
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
