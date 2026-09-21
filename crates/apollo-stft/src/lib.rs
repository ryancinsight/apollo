#![warn(missing_docs)]
//! Short-time Fourier transform.

/// Application-layer orchestration and execution plans.
pub mod application;
/// Domain contracts and error types.
pub mod domain;
/// CPU transport infrastructure.
pub mod infrastructure;

pub use application::execution::kernel::peak::{estimate_peaks, PeakEstimate};
pub use application::execution::kernel::window::Window;
pub use application::execution::plan::stft::dimension_1d::{is_valid_length, StftPlan};
pub use domain::contracts::error::{PeakEstimationError, StftError};
pub use infrastructure::transport::cpu::{istft, istft_leto, stft, stft_leto};

/// The crate README's examples, compiled and run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

/// GPU-accelerated backend using typed Hephaestus dispatch.
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
