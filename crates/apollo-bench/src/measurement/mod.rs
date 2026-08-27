mod calibration;
mod sample;

use crate::config::{BenchmarkConfig, BenchmarkMode};

pub(crate) use sample::SampleSummary;
use std::time::{Duration, Instant};

const PICOSECONDS_PER_NANOSECOND: u128 = 1_000;

pub(crate) fn measure(config: BenchmarkConfig, mut operation: impl FnMut()) -> SampleSummary {
    if config.mode() == BenchmarkMode::Smoke {
        let sample_start = Instant::now();
        operation();
        return SampleSummary::from_single_observation(normalized_picoseconds(
            sample_start.elapsed(),
            1,
        ));
    }

    let warm_up_start = Instant::now();
    let mut warm_up_iterations = 0_u64;
    while warm_up_start.elapsed() < config.warm_up() {
        operation();
        warm_up_iterations = warm_up_iterations.saturating_add(1);
    }

    let iterations_per_sample = calibration::iterations_per_sample(
        config.measurement().as_nanos(),
        config.sample_count(),
        warm_up_start.elapsed().as_nanos(),
        warm_up_iterations,
    );
    let sample_count = config.sample_count().get();
    let mut samples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let sample_start = Instant::now();
        for _ in 0..iterations_per_sample {
            operation();
        }
        samples.push(normalized_picoseconds(
            sample_start.elapsed(),
            iterations_per_sample,
        ));
    }

    SampleSummary::from_samples(samples, iterations_per_sample)
        .expect("invariant: the non-zero sample count fills every timing sample")
}

fn normalized_picoseconds(elapsed: Duration, iterations: u64) -> u128 {
    debug_assert!(
        iterations > 0,
        "calibration always returns at least one iteration"
    );
    elapsed.as_nanos() * PICOSECONDS_PER_NANOSECOND / u128::from(iterations)
}

#[cfg(test)]
mod tests {
    use super::normalized_picoseconds;
    use std::time::Duration;

    #[test]
    fn normalization_preserves_sub_nanosecond_observations() {
        assert_eq!(normalized_picoseconds(Duration::from_nanos(1), 4), 250);
    }
}
