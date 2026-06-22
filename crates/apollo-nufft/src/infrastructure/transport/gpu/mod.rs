#![warn(missing_docs)]

//! WGPU NUFFT backend boundary for Apollo.
//!
//! This crate owns NUFFT-specific WGPU capability and plan descriptors. Dense
//! FFT WGPU execution lives in `apollo-fft-wgpu`; CPU NUFFT math and metadata
//! live in `apollo-nufft`.
//!
//! # Features
//!
//! - `debug-readbacks`: Enable intermediate GPU grid readbacks after load and IFFT
//!   stages for numerical triage. Intended for testing only; adds GPU→CPU
//!   synchronization points.

/// Application-layer NUFFT WGPU plan descriptors.
pub mod application;
/// Domain contracts for WGPU NUFFT execution.
pub mod domain;
/// Infrastructure boundary for WGPU device acquisition.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
pub use domain::capabilities::NufftWgpuCapabilities;
pub use domain::error::{NufftWgpuError, NufftWgpuResult};
pub use infrastructure::device::{nufft_wgpu_available, NufftWgpuBackend};
pub use infrastructure::{NufftGpuBuffers1D, NufftGpuBuffers3D};
#[cfg(any(test, feature = "diagnostics"))]
pub use infrastructure::{NufftGridSnapshot, NufftType2GridDiagnostics};
