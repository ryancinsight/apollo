mod text;

use crate::case::BenchmarkCase;
use crate::measurement::SampleSummary;

/// Immutable result of measuring one benchmark case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkRecord {
    case: BenchmarkCase,
    minimum_nanoseconds: u128,
    median_nanoseconds: u128,
    sample_count: usize,
    iterations_per_sample: u64,
}

impl BenchmarkRecord {
    pub(crate) fn new(case: BenchmarkCase, summary: SampleSummary) -> Self {
        Self {
            case,
            minimum_nanoseconds: summary.minimum_nanoseconds,
            median_nanoseconds: summary.median_nanoseconds,
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
