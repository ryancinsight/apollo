#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Native benchmark measurement for Apollo's CPU and provider-backed kernels.
//!
//! # Estimator assumptions
//!
//! Let `x₁, …, x₂m` be the 100 per-operation timing samples and let the
//! reported value be `⌊(xₘ + x₍m₊₁₎) / 2⌋` after sorting. Replacing fewer
//! than `m` observations cannot move this value outside the original sample
//! range: more than half the observations remain inside that range. It can
//! still change both central ranks. For example, replacing `1` in `1..=100`
//! by a larger-than-100 observation changes the reported median from 50 to 51.
//!
//! This finite replacement bound does not establish timing stability. Neither
//! sorting nor counterbalancing removes serial dependence or systematic drift.
//!
//! For a predetermined comparison family containing `c` cases, the comparator
//! selects each baseline and candidate interval with miscoverage at most `0.05 / (2c)`,
//! assuming independent observations from one fixed distribution per interval.
//! Integer timing ties make the binomial coverage a lower bound, not an exact
//! coverage probability. Bonferroni's inequality bounds joint miscoverage by
//! 5% without independence *between intervals*; it does not remove the
//! within-interval sampling assumption. Intersecting four comparison events
//! preserves that bound under the same assumptions, not under arbitrary host
//! drift. These conditions are not established by the report schema. Discard
//! measurements invalidated by unchanged-control drift as performance evidence.
//! The interval construction follows
//! [NIST Technical Note 2119, section 5.3](https://doi.org/10.6028/NIST.TN.2119).
//!
//! Benchmark closures execute sequentially. Parallel execution would overlap
//! the measured work and destroy the per-operation timing contract; Moirai
//! remains the provider for transform runtime parallelism.

mod case;
mod comparison;
mod config;
mod error;
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
pub use config::{BenchmarkConfig, BenchmarkConfigError, BenchmarkMode, BenchmarkModeError};
pub use error::BenchmarkError;
pub use measurement::{
    bind_measurement_processor, MeasurementProcessor, ProcessorSelection, ProcessorSelectionError,
    PROCESSOR_VAR,
};
pub use report::BenchmarkRecord;
pub use suite::BenchmarkSuite;
