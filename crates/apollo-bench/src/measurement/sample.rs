/// Summarizes one case's normalized wall-clock samples.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SampleSummary {
    pub(crate) minimum_nanoseconds: u128,
    pub(crate) median_nanoseconds: u128,
    pub(crate) median_lower_nanoseconds: u128,
    pub(crate) median_upper_nanoseconds: u128,
    pub(crate) median_confidence_parts_per_million: u32,
    pub(crate) sample_count: usize,
    pub(crate) iterations_per_sample: u64,
}

impl SampleSummary {
    pub(crate) fn from_samples(mut samples: Vec<u128>, iterations_per_sample: u64) -> Option<Self> {
        samples.sort_unstable();
        let lower = *samples.get((samples.len().checked_sub(1)?) / 2)?;
        let upper = *samples.get(samples.len() / 2)?;
        let median_interval = MedianInterval::from_sorted_samples(&samples)?;
        Some(Self {
            minimum_nanoseconds: *samples.first()?,
            median_nanoseconds: lower + (upper - lower) / 2,
            median_lower_nanoseconds: median_interval.lower_nanoseconds,
            median_upper_nanoseconds: median_interval.upper_nanoseconds,
            median_confidence_parts_per_million: median_interval.confidence_parts_per_million,
            sample_count: samples.len(),
            iterations_per_sample,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MedianInterval {
    lower_nanoseconds: u128,
    upper_nanoseconds: u128,
    confidence_parts_per_million: u32,
}

impl MedianInterval {
    /// Computes the narrowest symmetric nonparametric interval whose coverage
    /// is at least 95%.
    ///
    /// For ordered samples `X_(1), …, X_(n)`, the interval
    /// `[X_(k), X_(n-k+1)]` covers the population median with probability
    /// `1 - 2 * P(Bin(n, 0.5) <= k - 1)`. Apollo fixes `n = 100`, so the
    /// binomial counts and parts-per-million conversion fit exactly in
    /// `u128`. See NIST Technical Note 2119, section 5.3, equations 30–31:
    /// <https://doi.org/10.6028/NIST.TN.2119>.
    fn from_sorted_samples(samples: &[u128]) -> Option<Self> {
        const TARGET_NUMERATOR: u128 = 95;
        const TARGET_DENOMINATOR: u128 = 100;
        const PARTS_PER_MILLION: u128 = 1_000_000;
        const MAX_EXACT_SAMPLE_COUNT: usize = 100;

        let sample_count = samples.len();
        if sample_count == 0 || sample_count > MAX_EXACT_SAMPLE_COUNT {
            return None;
        }

        let total_outcomes = 1_u128 << sample_count;
        let mut binomial_coefficient = 1_u128;
        let mut lower_tail_outcomes = 1_u128;
        let mut selected_rank = 1_usize;
        let mut selected_coverage = total_outcomes.saturating_sub(2);

        for rank in 2..=(sample_count + 1) / 2 {
            let previous_successes = rank - 2;
            binomial_coefficient = binomial_coefficient
                * u128::try_from(sample_count - previous_successes)
                    .expect("invariant: Apollo sample counts fit in u128")
                / u128::try_from(previous_successes + 1)
                    .expect("invariant: positive binomial divisor fits in u128");
            lower_tail_outcomes += binomial_coefficient;
            let coverage_outcomes = total_outcomes.saturating_sub(2 * lower_tail_outcomes);
            if coverage_outcomes * TARGET_DENOMINATOR < total_outcomes * TARGET_NUMERATOR {
                break;
            }
            selected_rank = rank;
            selected_coverage = coverage_outcomes;
        }

        let confidence_parts_per_million =
            u32::try_from(selected_coverage * PARTS_PER_MILLION / total_outcomes)
                .expect("invariant: confidence is bounded by one million parts");
        Some(Self {
            lower_nanoseconds: samples[selected_rank - 1],
            upper_nanoseconds: samples[sample_count - selected_rank],
            confidence_parts_per_million,
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
        assert_eq!(summary.median_lower_nanoseconds, 2);
        assert_eq!(summary.median_upper_nanoseconds, 1_000_000);
        assert_eq!(summary.median_confidence_parts_per_million, 937_500);
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

    #[test]
    fn standard_sample_count_has_exact_distribution_free_median_bounds() {
        let summary = SampleSummary::from_samples((1..=100).collect(), 1)
            .expect("invariant: the standard sample set is non-empty");

        assert_eq!(summary.median_lower_nanoseconds, 40);
        assert_eq!(summary.median_upper_nanoseconds, 61);
        assert_eq!(summary.median_confidence_parts_per_million, 964_799);
    }
}
