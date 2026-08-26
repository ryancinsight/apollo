mod text;

use crate::case::BenchmarkCase;
use crate::measurement::SampleSummary;

/// Immutable result of measuring one benchmark case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkRecord {
    case: BenchmarkCase,
    minimum_picoseconds: u128,
    median_picoseconds: u128,
    median_lower_picoseconds: u128,
    median_upper_picoseconds: u128,
    median_confidence_parts_per_million: u32,
    ordered_samples_picoseconds: Box<[u128]>,
    iterations_per_sample: u64,
}

impl BenchmarkRecord {
    pub(crate) fn new(case: BenchmarkCase, summary: SampleSummary) -> Self {
        Self {
            case,
            minimum_picoseconds: summary.minimum_picoseconds,
            median_picoseconds: summary.median_picoseconds,
            median_lower_picoseconds: summary.median_lower_picoseconds,
            median_upper_picoseconds: summary.median_upper_picoseconds,
            median_confidence_parts_per_million: summary.median_confidence_parts_per_million,
            ordered_samples_picoseconds: summary.ordered_samples_picoseconds,
            iterations_per_sample: summary.iterations_per_sample,
        }
    }

    /// Returns the minimum normalized duration in picoseconds.
    #[must_use]
    pub const fn minimum_picoseconds(&self) -> u128 {
        self.minimum_picoseconds
    }

    /// Returns the median normalized duration in picoseconds.
    #[must_use]
    pub const fn median_picoseconds(&self) -> u128 {
        self.median_picoseconds
    }

    /// Returns the lower order statistic of the median confidence interval.
    #[must_use]
    pub const fn median_lower_picoseconds(&self) -> u128 {
        self.median_lower_picoseconds
    }

    /// Returns the upper order statistic of the median confidence interval.
    #[must_use]
    pub const fn median_upper_picoseconds(&self) -> u128 {
        self.median_upper_picoseconds
    }

    /// Returns the exact median interval coverage in parts per million.
    #[must_use]
    pub const fn median_confidence_parts_per_million(&self) -> u32 {
        self.median_confidence_parts_per_million
    }

    /// Returns the number of timing samples.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.ordered_samples_picoseconds.len()
    }

    /// Returns the normalized timing samples in nondecreasing order.
    #[must_use]
    pub fn ordered_samples_picoseconds(&self) -> &[u128] {
        &self.ordered_samples_picoseconds
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
