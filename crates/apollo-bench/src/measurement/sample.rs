/// Summarizes one case's normalized wall-clock samples.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SampleSummary {
    pub(crate) minimum_nanoseconds: u128,
    pub(crate) median_nanoseconds: u128,
    pub(crate) sample_count: usize,
    pub(crate) iterations_per_sample: u64,
}

impl SampleSummary {
    pub(crate) fn from_samples(mut samples: Vec<u128>, iterations_per_sample: u64) -> Option<Self> {
        samples.sort_unstable();
        let lower = *samples.get((samples.len().checked_sub(1)?) / 2)?;
        let upper = *samples.get(samples.len() / 2)?;
        Some(Self {
            minimum_nanoseconds: *samples.first()?,
            median_nanoseconds: lower + (upper - lower) / 2,
            sample_count: samples.len(),
            iterations_per_sample,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SampleSummary;

    #[test]
    fn median_resists_a_single_large_outlier() {
        let summary = SampleSummary::from_samples(vec![3, 5, 4, 1_000_000, 2], 1)
            .expect("invariant: literal sample set is non-empty");
        assert_eq!(summary.minimum_nanoseconds, 2);
        assert_eq!(summary.median_nanoseconds, 4);
        assert_eq!(summary.sample_count, 5);
    }

    #[test]
    fn even_sample_median_averages_the_central_pair() {
        let summary = SampleSummary::from_samples(vec![10, 2, 7, 4], 1)
            .expect("invariant: literal sample set is non-empty");
        assert_eq!(summary.median_nanoseconds, 5);
    }

    #[test]
    fn empty_samples_do_not_invent_a_timing_summary() {
        assert_eq!(SampleSummary::from_samples(Vec::new(), 1), None);
    }
}
