/// Symmetric order-statistic interval with binomial coverage bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MedianInterval {
    pub(crate) lower_picoseconds: u128,
    pub(crate) upper_picoseconds: u128,
    pub(crate) confidence_parts_per_million: u32,
}

impl MedianInterval {
    /// Computes the narrowest symmetric interval that controls joint
    /// miscoverage at 5% over `simultaneous_intervals` intervals.
    ///
    /// For independent samples from one fixed distribution, sorted as
    /// `X_(1), …, X_(n)`, the interval `[X_(k), X_(n-k+1)]` covers a
    /// population median with probability at least
    /// `1 - 2 * P(Bin(n, 0.5) <= k - 1)`. Equality holds without probability
    /// mass at the median; discrete timing ties can increase coverage.
    /// Bonferroni's inequality requires each interval's miscoverage to be at most
    /// `0.05 / simultaneous_intervals`. Apollo fixes `n = 100`, so binomial
    /// outcome counts fit exactly in `u128`. The reported parts per million
    /// floor this bound. Dependence between intervals is permitted, but this
    /// function cannot establish independence or stationarity of observations.
    /// Sample count and comparison family must be chosen independently of
    /// observed timings; adaptive selection requires a separate error analysis.
    ///
    /// Median coverage: NIST Technical Note 2119, section 5.3, equations
    /// 30–31: <https://doi.org/10.6028/NIST.TN.2119>.
    ///
    /// Family-wise coverage: NIST/SEMATECH Handbook, section 7.4.6.3:
    /// <https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm>.
    pub(crate) fn from_ordered_samples(
        samples: &[u128],
        simultaneous_intervals: usize,
    ) -> Option<Self> {
        const FAMILY_ERROR_DENOMINATOR: u128 = 20;
        const PARTS_PER_MILLION: u128 = 1_000_000;
        const MAX_EXACT_SAMPLE_COUNT: usize = 100;

        let sample_count = samples.len();
        if sample_count == 0 || sample_count > MAX_EXACT_SAMPLE_COUNT || simultaneous_intervals == 0
        {
            return None;
        }

        let simultaneous_intervals = u128::try_from(simultaneous_intervals).ok()?;
        let total_outcomes = 1_u128 << sample_count;
        let mut binomial_coefficient = 1_u128;
        let mut lower_tail_outcomes = 1_u128;
        let mut selected_rank = 1_usize;
        let mut selected_coverage = total_outcomes.saturating_sub(2);

        if !meets_familywise_target(
            total_outcomes,
            lower_tail_outcomes,
            simultaneous_intervals,
            FAMILY_ERROR_DENOMINATOR,
        ) {
            return None;
        }

        for rank in 2..=(sample_count + 1) / 2 {
            let previous_successes = rank - 2;
            binomial_coefficient = binomial_coefficient
                * u128::try_from(sample_count - previous_successes).ok()?
                / u128::try_from(previous_successes + 1).ok()?;
            lower_tail_outcomes += binomial_coefficient;
            if !meets_familywise_target(
                total_outcomes,
                lower_tail_outcomes,
                simultaneous_intervals,
                FAMILY_ERROR_DENOMINATOR,
            ) {
                break;
            }
            selected_rank = rank;
            selected_coverage = total_outcomes.saturating_sub(2 * lower_tail_outcomes);
        }

        let confidence_parts_per_million =
            u32::try_from(selected_coverage * PARTS_PER_MILLION / total_outcomes).ok()?;
        Some(Self {
            lower_picoseconds: samples[selected_rank - 1],
            upper_picoseconds: samples[sample_count - selected_rank],
            confidence_parts_per_million,
        })
    }
}

fn meets_familywise_target(
    total_outcomes: u128,
    lower_tail_outcomes: u128,
    simultaneous_intervals: u128,
    error_denominator: u128,
) -> bool {
    lower_tail_outcomes
        .checked_mul(2)
        .and_then(|excluded| excluded.checked_mul(error_denominator))
        .and_then(|excluded| excluded.checked_mul(simultaneous_intervals))
        .is_some_and(|excluded| excluded <= total_outcomes)
}

#[cfg(test)]
mod tests {
    use super::MedianInterval;

    #[test]
    fn one_case_matches_the_exact_95_percent_interval() {
        let interval = MedianInterval::from_ordered_samples(&(1..=100).collect::<Vec<_>>(), 1)
            .expect("invariant: 100 samples support one-case coverage");

        assert_eq!(interval.lower_picoseconds, 40);
        assert_eq!(interval.upper_picoseconds, 61);
        assert_eq!(interval.confidence_parts_per_million, 964_799);
    }

    #[test]
    fn paired_comparison_accounts_for_both_intervals() {
        let samples = (1..=100).collect::<Vec<_>>();
        let paired = MedianInterval::from_ordered_samples(&samples, 2)
            .expect("invariant: 100 samples support a paired comparison");
        let two_cases = MedianInterval::from_ordered_samples(&samples, 4)
            .expect("invariant: 100 samples support two paired comparisons");

        assert_eq!(paired.lower_picoseconds, 39);
        assert_eq!(paired.upper_picoseconds, 62);
        assert_eq!(two_cases.lower_picoseconds, 38);
        assert_eq!(two_cases.upper_picoseconds, 63);
    }

    #[test]
    fn larger_simultaneous_family_selects_strictly_wider_bounds() {
        let samples = (1..=100).collect::<Vec<_>>();
        let one = MedianInterval::from_ordered_samples(&samples, 1)
            .expect("invariant: 100 samples support one-case coverage");
        let family = MedianInterval::from_ordered_samples(&samples, 400)
            .expect("invariant: 100 samples support 400-case coverage");

        assert!(family.lower_picoseconds < one.lower_picoseconds);
        assert!(family.upper_picoseconds > one.upper_picoseconds);
        assert!(family.confidence_parts_per_million > one.confidence_parts_per_million);
    }

    #[test]
    fn empty_samples_or_family_produce_no_interval() {
        assert_eq!(MedianInterval::from_ordered_samples(&[], 1), None);
        assert_eq!(MedianInterval::from_ordered_samples(&[1], 0), None);
    }

    #[test]
    fn independent_binary_samples_match_binomial_coverage() {
        // Six independent fair binary observations have 64 equiprobable
        // outcomes. The interval covers the population median 1 unless all
        // observations equal 0 or all equal 2: coverage is exactly 62/64.
        let mut covering = 0_u32;
        for outcome in 0..64_u32 {
            let mut samples: [u128; 6] =
                core::array::from_fn(|bit| u128::from((outcome >> bit) & 1) * 2);
            samples.sort_unstable();
            let interval = MedianInterval::from_ordered_samples(&samples, 1)
                .expect("invariant: six samples support the one-case interval");
            covering +=
                u32::from(interval.lower_picoseconds <= 1 && interval.upper_picoseconds >= 1);
            assert_eq!(interval.confidence_parts_per_million, 968_750);
        }
        assert_eq!(covering, 62);
    }

    #[test]
    fn perfectly_dependent_samples_do_not_inherit_binomial_coverage() {
        // One fair draw copied six times has the same marginal distribution
        // as above, but neither possible interval covers its median 1.
        let mut covering = 0_u32;
        for value in [0, 2] {
            let interval = MedianInterval::from_ordered_samples(&[value; 6], 1)
                .expect("invariant: six samples support the one-case interval");
            assert_eq!(interval.lower_picoseconds, value);
            assert_eq!(interval.upper_picoseconds, value);
            covering +=
                u32::from(interval.lower_picoseconds <= 1 && interval.upper_picoseconds >= 1);
            assert_eq!(interval.confidence_parts_per_million, 968_750);
        }
        assert_eq!(covering, 0);
    }

    #[test]
    fn point_mass_retains_a_conservative_coverage_bound() {
        // A constant population's only sample covers its median with
        // probability one. Report the floored binomial bound, not that
        // population-specific coverage, which the observations cannot identify.
        let interval = MedianInterval::from_ordered_samples(&[17; 100], 1)
            .expect("invariant: 100 observations support the one-case interval");
        assert_eq!(interval.lower_picoseconds, 17);
        assert_eq!(interval.upper_picoseconds, 17);
        assert_eq!(interval.confidence_parts_per_million, 964_799);
    }
}
