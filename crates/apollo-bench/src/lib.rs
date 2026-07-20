#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Native benchmark measurement for Apollo's CPU and provider-backed kernels.
//!
//! # Estimator theorem
//!
//! Let `x₁, …, x₂m` be the 100 per-operation timing samples and let the
//! reported value be `⌊(xₘ + x₍m₊₁₎) / 2⌋` after sorting. At least `m` samples
//! are no greater than `xₘ`, and at least `m` samples are no smaller than
//! `x₍m₊₁₎`. Therefore fewer than `m` arbitrarily large scheduler or
//! device-delay outliers cannot replace the central pair that determines the
//! reported median.
//!
//! The proof follows directly from sorted-index cardinality. This is a robust
//! summary property, not a claim that wall-clock measurements are noise-free
//! or comparable across machines.
//!
//! Benchmark closures execute sequentially. Parallel execution would overlap
//! the measured work and destroy the per-operation timing contract; Moirai
//! remains the provider for transform runtime parallelism.

mod case;
mod comparison;
mod config;
mod measurement;
mod report;
mod suite;

pub use case::BenchmarkCase;
pub use comparison::{
    compare_report_directories, BenchmarkRegression, ComparisonError, ComparisonSummary,
};
pub use config::{BenchmarkConfig, BenchmarkConfigError};
pub use report::BenchmarkRecord;
pub use suite::BenchmarkSuite;
