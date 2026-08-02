#![warn(missing_docs)]
//! Mellin transform plans for Apollo.
//!
//! The Mellin transform maps a positive scale-domain signal `f(r)` to
//! moments `M(s) = int_a^b f(r) r^(s-1) dr`. Under the logarithmic change of
//! variables `r = exp(u)`, the imaginary-axis Mellin transform becomes a
//! Fourier transform over `u`, which is the source of scale-shift behavior used
//! in scale-invariant matching.
//!
//! On a uniform log grid, forward scaling by `du` and inverse scaling by
//! `1 / (N * du)` form an exact DFT inverse pair in exact arithmetic. The
//! optional accelerator uses typed Hephaestus command streams over that same
//! contract, with a concrete `f32` scale grid and sealed host storage.
//!
//! This crate owns positive scale-domain contracts, log-resampling, real
//! Mellin moments, log-frequency Mellin spectra, and value-semantic tests.

/// Application-layer Mellin plans.
pub mod application;
/// Domain contracts and metadata.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
#[cfg(test)]
mod verification;

pub use application::execution::plan::mellin::{
    MellinGpuStorage, MellinPlan, MellinSpectrum, MellinStorage,
};
pub use domain::contracts::error::{MellinError, MellinResult};
pub use domain::metadata::scale::MellinScaleConfig;
pub use infrastructure::kernel::resample::{
    calculate_log_resample, exp_resample, inverse_log_frequency_spectrum, log_frequency_spectrum,
    mellin_moment,
};

#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
