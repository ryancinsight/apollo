mod text;

use crate::case::BenchmarkCase;
use crate::measurement::SampleSummary;

/// Immutable result of measuring one benchmark case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkRecord {
    case: BenchmarkCase,
    minimum_nanoseconds: u128,
    median_nanoseconds: u128,
    median_lower_nanoseconds: u128,
    median_upper_nanoseconds: u128,
    median_confidence_parts_per_million: u32,
    sample_count: usize,
    iterations_per_sample: u64,
}

impl BenchmarkRecord {
    pub(crate) fn new(case: BenchmarkCase, summary: SampleSummary) -> Self {
        Self {
            case,
            minimum_nanoseconds: summary.minimum_nanoseconds,
            median_nanoseconds: summary.median_nanoseconds,
            median_lower_nanoseconds: summary.median_lower_nanoseconds,
            median_upper_nanoseconds: summary.median_upper_nanoseconds,
            median_confidence_parts_per_million: summary.median_confidence_parts_per_million,
            sample_count: summary.sample_count,
            iterations_per_sample: summary.iterations_per_sample,
        }
    }

    /// Returns the minimum normalized duration in nanoseconds.
    #[must_use]
    pub const fn minimum_nanoseconds(&self) -> u128 {
        self.minimum_nanoseconds
    }

    /// Returns the median normalized duration in nanoseconds.
    #[must_use]
    pub const fn median_nanoseconds(&self) -> u128 {
        self.median_nanoseconds
    }

    /// Returns the lower order statistic of the median confidence interval.
    #[must_use]
    pub const fn median_lower_nanoseconds(&self) -> u128 {
        self.median_lower_nanoseconds
    }

    /// Returns the upper order statistic of the median confidence interval.
    #[must_use]
    pub const fn median_upper_nanoseconds(&self) -> u128 {
        self.median_upper_nanoseconds
    }

    /// Returns the exact median interval coverage in parts per million.
    #[must_use]
    pub const fn median_confidence_parts_per_million(&self) -> u32 {
        self.median_confidence_parts_per_million
    }

    /// Returns the number of timing samples.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns production-closure iterations in each timing sample.
    #[must_use]
    pub const fn iterations_per_sample(&self) -> u64 {
        self.iterations_per_sample
    }
}

pub(crate) fn render(records: &[BenchmarkRecord]) -> String {
    text::render(records)
}
