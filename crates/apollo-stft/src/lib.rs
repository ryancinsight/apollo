#![warn(missing_docs)]
//! Short-time Fourier transform.

/// Application-layer orchestration and execution plans.
pub mod application;
/// Domain contracts and error types.
pub mod domain;
/// CPU transport infrastructure.
pub mod infrastructure;

pub use application::execution::plan::stft::dimension_1d::{is_valid_length, StftPlan};
pub use application::execution::plan::stft::storage::{
    StftRealOutputStorage, StftRealStorage, StftSpectrumInput, StftSpectrumStorage,
};
pub use domain::contracts::error::StftError;
pub use infrastructure::transport::cpu::{istft, istft_leto, stft, stft_leto};

/// GPU-accelerated backend using WGPU.
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    pub use crate::infrastructure::transport::gpu::*;
}
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
