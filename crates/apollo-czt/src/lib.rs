#![warn(missing_docs)]
//! Chirp z-transform utilities for Apollo.
//!
//! This crate implements the chirp z-transform as a reusable plan plus direct
//! reference execution and a Bluestein-style fast path for arbitrary sample
//! lengths.
//!
//! Architectural alignment preserves standard domain boundary decoupling.

/// Execution structures and caching rules.
pub mod application;
/// Mathematical boundary schemas.
pub mod domain;
/// CPU execution helpers.
pub mod infrastructure;

pub use application::execution::plan::czt::dimension_1d::{CztPlan, CztStorage};
pub use domain::contracts::error::CztError;
pub use domain::metadata::CztParameters;
pub use infrastructure::transport::cpu::{czt, czt_direct, czt_direct_leto, czt_leto};

/// GPU-accelerated backend using WGPU.
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    pub use crate::infrastructure::transport::gpu::*;
}
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
