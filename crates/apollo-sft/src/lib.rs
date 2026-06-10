#![warn(missing_docs)]
//! Sparse Fourier transform plans and utilities for Apollo.
//!
//! `apollo-sft` is the single source of truth for sparse Fourier transforms.
//! Dense FFT infrastructure remains in `apollo-fft`; sparse spectrum domain
//! modeling, sparse plan configuration, coefficient selection, and sparse
//! reconstruction live here.

/// Application-layer sparse transform execution.
pub mod application;
/// Domain types and contracts.
pub mod domain;
/// Infrastructure kernels used by the sparse transform implementation.
pub mod infrastructure;

#[cfg(test)]
mod verification;

pub use application::execution::transform::sparse::{
    SparseComplexStorage, SparseFftPlan, SparseLetoSpectrum,
};
pub use domain::spectrum::sparse::SparseSpectrum;
