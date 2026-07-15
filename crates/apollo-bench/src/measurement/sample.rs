/// Summarizes one case's normalized wall-clock samples.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SampleSummary {
    pub(crate) minimum_nanoseconds: u128,
    pub(crate) median_nanoseconds: u128,
    pub(crate) sample_count: usize,
    pub(crate) iterations_per_sample: u64,
}

impl SampleSummary {
    pub(crate) fn from_samples(mut samples: Vec<u128>, iterations_per_sample: u64) -> Self {
        samples.sort_unstable();
        let middle = samples.len() / 2;
        Self {
            minimum_nanoseconds: samples[0],
            median_nanoseconds: samples[middle],
            sample_count: samples.len(),
            iterations_per_sample,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SampleSummary;

    #[test]
    fn median_resists_a_single_large_outlier() {
        let summary = SampleSummary::from_samples(vec![3, 5, 4, 1_000_000, 2], 1);
        assert_eq!(summary.minimum_nanoseconds, 2);
        assert_eq!(summary.median_nanoseconds, 4);
        assert_eq!(summary.sample_count, 5);
    }
}
