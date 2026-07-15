//! Native benchmark measurement for Apollo's CPU and provider-backed kernels.
//!
//! # Estimator theorem
//!
//! Let `x₁, …, x₂m₊₁` be per-operation timing samples and let `x₍m₊₁₎` be
//! their sorted middle element. At least `m + 1` samples are no smaller than
//! `x₍m₊₁₎`, and at least `m + 1` samples are no greater. Therefore a minority
//! of arbitrarily large scheduler or device-delay outliers cannot move the
//! reported median outside the central order statistic.
//!
//! The proof follows directly from sorted-index cardinality. This is a robust
//! summary property, not a claim that wall-clock measurements are noise-free
//! or comparable across machines.
//!
//! Benchmark closures execute sequentially. Parallel execution would overlap
//! the measured work and destroy the per-operation timing contract; Moirai
//! remains the provider for transform runtime parallelism.

mod case;
mod config;
mod measurement;
mod report;
mod suite;

pub use case::BenchmarkCase;
pub use config::{BenchmarkConfig, BenchmarkConfigError};
pub use report::BenchmarkRecord;
pub use suite::BenchmarkSuite;
