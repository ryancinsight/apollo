#![warn(missing_docs)]
//! Fast Walsh-Hadamard transform utilities for Apollo.
//!
//! This crate implements the unnormalized Hadamard transform on vectors whose
//! length is a power of two. The transform evaluates across threads for limits > 1024.
//!
//! Architectural alignment preserves standard domain boundary decoupling.

/// Execution structures and caching rules.
pub mod application;
/// Mathematical boundary schemas.
pub mod domain;
/// CPU execution helpers.
pub mod infrastructure;

pub use application::execution::plan::fwht::dimension_1d::FwhtPlan;
pub use application::execution::plan::fwht::dimension_2d::FwhtPlan2D;
pub use application::execution::plan::fwht::dimension_3d::FwhtPlan3D;
pub use application::execution::plan::fwht::storage::FwhtStorage;
pub use domain::contracts::error::FwhtError;
pub use infrastructure::transport::cpu::{fwht, fwht_leto, ifwht, ifwht_leto};
