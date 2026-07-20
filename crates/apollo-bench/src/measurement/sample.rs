/// Summarizes one case's normalized wall-clock samples.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SampleSummary {
    pub(crate) minimum_nanoseconds: u128,
    pub(crate) median_nanoseconds: u128,
    pub(crate) median_lower_nanoseconds: u128,
    pub(crate) median_upper_nanoseconds: u128,
    pub(crate) median_confidence_parts_per_million: u32,
    pub(crate) ordered_samples_nanoseconds: Box<[u128]>,
    pub(crate) iterations_per_sample: u64,
}

impl SampleSummary {
    pub(crate) fn from_samples(mut samples: Vec<u128>, iterations_per_sample: u64) -> Option<Self> {
        samples.sort_unstable();
        let lower = *samples.get((samples.len().checked_sub(1)?) / 2)?;
        let upper = *samples.get(samples.len() / 2)?;
        let median_interval =
            crate::statistics::median::MedianInterval::from_ordered_samples(&samples, 1)?;
        Some(Self {
            minimum_nanoseconds: *samples.first()?,
            median_nanoseconds: lower + (upper - lower) / 2,
            median_lower_nanoseconds: median_interval.lower_nanoseconds,
            median_upper_nanoseconds: median_interval.upper_nanoseconds,
            median_confidence_parts_per_million: median_interval.confidence_parts_per_million,
            ordered_samples_nanoseconds: samples.into_boxed_slice(),
            iterations_per_sample,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SampleSummary;

    #[test]
    fn median_resists_a_single_large_outlier() {
        let summary = SampleSummary::from_samples(vec![1, 2, 3, 4, 5, 1_000_000], 1)
            .expect("invariant: six samples support the one-case interval");
        assert_eq!(summary.minimum_nanoseconds, 1);
        assert_eq!(summary.median_nanoseconds, 3);
        assert_eq!(summary.median_lower_nanoseconds, 1);
        assert_eq!(summary.median_upper_nanoseconds, 1_000_000);
        assert_eq!(summary.median_confidence_parts_per_million, 968_750);
        assert_eq!(summary.ordered_samples_nanoseconds.len(), 6);
    }

    #[test]
    fn even_sample_median_averages_the_central_pair() {
        let summary = SampleSummary::from_samples(vec![10, 2, 7, 4, 8, 6], 1)
            .expect("invariant: six samples support the one-case interval");
        assert_eq!(summary.median_nanoseconds, 6);
    }

    #[test]
    fn insufficient_samples_do_not_invent_a_timing_summary() {
        assert_eq!(SampleSummary::from_samples(Vec::new(), 1), None);
        assert_eq!(SampleSummary::from_samples(vec![1, 2, 3, 4, 5], 1), None);
    }

    #[test]
    fn standard_sample_count_has_exact_distribution_free_median_bounds() {
        let summary = SampleSummary::from_samples((1..=100).collect(), 1)
            .expect("invariant: the standard sample set is non-empty");

        assert_eq!(summary.median_lower_nanoseconds, 40);
        assert_eq!(summary.median_upper_nanoseconds, 61);
        assert_eq!(summary.median_confidence_parts_per_million, 964_799);
    }
}
