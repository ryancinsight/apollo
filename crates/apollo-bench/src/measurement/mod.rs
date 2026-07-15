mod calibration;
mod sample;

use crate::config::BenchmarkConfig;

pub(crate) use sample::SampleSummary;
use std::time::Instant;

pub(crate) fn measure(config: BenchmarkConfig, mut operation: impl FnMut()) -> SampleSummary {
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
    let mut samples = Vec::with_capacity(config.sample_count());
    for _ in 0..config.sample_count() {
        let sample_start = Instant::now();
        for _ in 0..iterations_per_sample {
            operation();
        }
        samples.push(sample_start.elapsed().as_nanos() / u128::from(iterations_per_sample));
    }

    SampleSummary::from_samples(samples, iterations_per_sample)
}
