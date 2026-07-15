#![warn(missing_docs)]
//! Graph Fourier transform plans and utilities for Apollo.
//!
//! `apollo-gft` owns graph-domain validation, combinatorial Laplacian
//! construction, spectral basis construction, and graph Fourier plan execution.

/// Application-layer GFT plans.
pub mod application;
/// Domain contracts and graph descriptors.
pub mod domain;
/// Infrastructure kernels for graph spectral construction.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::gft::{GftGpuStorage, GftPlan, GftStorage};
pub use domain::contracts::error::{GftError, GftResult};
pub use domain::graph::adjacency::GraphAdjacency;

/// GPU-accelerated backend using WGPU.
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    pub use crate::infrastructure::transport::gpu::*;
}
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
