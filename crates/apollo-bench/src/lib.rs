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
//! For a comparison containing `m` cases, the native comparator selects each
//! baseline and candidate interval with miscoverage at most `0.05 / (2m)`.
//! Bonferroni's inequality then bounds the probability that any of the `2m`
//! intervals misses its population median by 5%, without an independence
//! assumption. CI intersects four such comparison events across phase-reversed
//! ABBA and BAAB blocks, so the final family-wise false-positive event remains
//! bounded by any one comparison's 5% bound.
//!
//! Benchmark closures execute sequentially. Parallel execution would overlap
//! the measured work and destroy the per-operation timing contract; Moirai
//! remains the provider for transform runtime parallelism.

mod case;
mod comparison;
mod config;
mod measurement;
mod report;
mod statistics;
mod suite;

pub use case::BenchmarkCase;
pub use comparison::{
    compare_counterbalanced_report_directories,
    compare_replicated_counterbalanced_report_directories, compare_report_directories,
    BenchmarkRegression, ComparisonError, ComparisonSummary, CounterbalancedBenchmarkRegression,
    CounterbalancedComparisonSummary, CounterbalancedReportSet, IntervalSeparation,
    ReplicatedCounterbalancedBenchmarkRegression, ReplicatedCounterbalancedComparisonSummary,
};
pub use config::{BenchmarkConfig, BenchmarkConfigError};
pub use report::BenchmarkRecord;
pub use suite::BenchmarkSuite;
