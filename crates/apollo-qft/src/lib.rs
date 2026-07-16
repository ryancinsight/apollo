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
#[cfg(test)]
mod verification;

pub use application::execution::plan::qft::{
    iqft, iqft_leto, qft, qft_leto, QftGpuStorage, QftPlan, QftStorage,
};
pub use domain::contracts::error::{QftError, QftResult};
pub use domain::state::dimension::{is_valid_length, QuantumStateDimension};

#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
