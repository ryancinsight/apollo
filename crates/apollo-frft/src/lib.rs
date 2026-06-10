#![warn(missing_docs)]
//! Fractional Fourier transform utilities for Apollo.
//!
//! This crate evaluates the continuous fractional Fourier kernel on a finite
//! uniform sample vector. The implementation is a direct reference plan: it
//! preserves the mathematical definition across the full phase
//! domain, and respects the Apollo deep hierarchical namespace structure.
//!
//! Two plans are available:
//! - [`FrftPlan`]: direct O(N²) chirp kernel; fast but non-unitary for
//!   non-integer orders.
//! - [`UnitaryFrftPlan`]: Candan (2000) eigendecomposition; O(N³) construction,
//!   O(N²) per call, provably unitary for all real orders.

/// Application entry points and configuration models for FrFT evaluation.
pub mod application;
/// Domain contracts for FrFT evaluation.
pub mod domain;
/// High-performance computing kernels and backend hardware integrations.
pub mod infrastructure;

// Canonical crate-root API for transform plans and error contracts.

pub use crate::application::execution::plan::frft::dimension_1d::{frft, frft_leto, FrftPlan};
pub use crate::application::execution::plan::frft::storage::FrftStorage;
pub use crate::application::execution::plan::frft::unitary::{GrunbaumBasis, UnitaryFrftPlan};
pub use crate::domain::contracts::error::FrftError;
