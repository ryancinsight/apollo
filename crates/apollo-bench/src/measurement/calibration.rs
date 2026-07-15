pub(crate) fn iterations_per_sample(
    measurement_nanoseconds: u128,
    sample_count: usize,
    warm_up_nanoseconds: u128,
    warm_up_iterations: u64,
) -> u64 {
    let sample_budget = ceil_div(measurement_nanoseconds, sample_count as u128);
    let elapsed = warm_up_nanoseconds.max(1);
    let estimate = ceil_div(
        sample_budget.saturating_mul(u128::from(warm_up_iterations.max(1))),
        elapsed,
    )
    .max(1);

    u64::try_from(estimate).unwrap_or(u64::MAX)
}

const fn ceil_div(dividend: u128, divisor: u128) -> u128 {
    let quotient = dividend / divisor;
    if dividend % divisor == 0 {
        quotient
    } else {
        quotient + 1
    }
}

#[cfg(test)]
mod tests {
    use super::iterations_per_sample;

    #[test]
    fn calibration_scales_a_sample_to_its_budget() {
        assert_eq!(iterations_per_sample(1_000, 10, 100, 2), 2);
        assert_eq!(iterations_per_sample(1_000, 10, 100, 1), 1);
        assert_eq!(iterations_per_sample(1_000, 10, 0, 0), 100);
    }
}
