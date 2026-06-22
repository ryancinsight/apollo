#![warn(missing_docs)]
//! Quantum Fourier transform plans and utilities for Apollo.
//!
//! `apollo-qft` owns quantum state-dimension validation, dense unitary QFT
//! kernel execution, reusable plans, and value-semantic verification.

/// Application-layer QFT plans.
pub mod application;
/// Domain contracts and state descriptors.
pub mod domain;
/// Infrastructure kernels.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::qft::{iqft, iqft_leto, qft, qft_leto, QftPlan, QftStorage};
pub use domain::contracts::error::{QftError, QftResult};
pub use domain::state::dimension::{is_valid_length, QuantumStateDimension};

/// GPU-accelerated backend using WGPU.
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    pub use crate::infrastructure::transport::gpu::*;
}
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::{*};
