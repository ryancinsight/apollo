mod calibration;
mod processor;
mod sample;

pub use processor::{
    bind_measurement_processor, MeasurementProcessor, ProcessorSelection, ProcessorSelectionError,
    PROCESSOR_VAR,
};

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

/// Measures two operations as an interleaved pair sharing one window.
///
/// Sequential per-arm windows alias with package-turbo oscillation: each
/// window randomly catches boost or decay, so cross-arm ratios swing with
/// turbo state instead of code (#639). This warms both arms up together,
/// calibrates each arm from its own warm-up rate, then alternates timed
/// sample blocks (A,B,B,A,…) so both arms' samples draw from the same
/// turbo trajectory and their medians are directly comparable. The lead
/// arm alternates per sample to counterbalance order effects.
///
/// Each arm still reports its own [`SampleSummary`] with its own
/// iterations-per-sample; the pairing is a sampling property, not a
/// schema change.
pub(crate) fn measure_paired(
    config: BenchmarkConfig,
    mut operation_a: impl FnMut(),
    mut operation_b: impl FnMut(),
) -> (SampleSummary, SampleSummary) {
    if config.mode() == BenchmarkMode::Smoke {
        let start_a = Instant::now();
        operation_a();
        let summary_a =
            SampleSummary::from_single_observation(normalized_picoseconds(start_a.elapsed(), 1));
        let start_b = Instant::now();
        operation_b();
        let summary_b =
            SampleSummary::from_single_observation(normalized_picoseconds(start_b.elapsed(), 1));
        return (summary_a, summary_b);
    }

    let warm_up_start = Instant::now();
    let mut iterations_a = 0_u64;
    let mut iterations_b = 0_u64;
    let mut nanos_a = 0_u128;
    let mut nanos_b = 0_u128;
    while warm_up_start.elapsed() < config.warm_up() {
        let start = Instant::now();
        operation_a();
        nanos_a += start.elapsed().as_nanos();
        iterations_a = iterations_a.saturating_add(1);
        let start = Instant::now();
        operation_b();
        nanos_b += start.elapsed().as_nanos();
        iterations_b = iterations_b.saturating_add(1);
    }

    let per_sample_a = calibration::iterations_per_sample(
        config.measurement().as_nanos(),
        config.sample_count(),
        nanos_a,
        iterations_a,
    );
    let per_sample_b = calibration::iterations_per_sample(
        config.measurement().as_nanos(),
        config.sample_count(),
        nanos_b,
        iterations_b,
    );
    let sample_count = config.sample_count().get();
    let mut samples_a = Vec::with_capacity(sample_count);
    let mut samples_b = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        // Counterbalanced lead: even samples run A first, odd samples B.
        let (sample_a, sample_b) = if index % 2 == 0 {
            (
                timed_block(&mut operation_a, per_sample_a),
                timed_block(&mut operation_b, per_sample_b),
            )
        } else {
            let sample_b = timed_block(&mut operation_b, per_sample_b);
            let sample_a = timed_block(&mut operation_a, per_sample_a);
            (sample_a, sample_b)
        };
        samples_a.push(sample_a);
        samples_b.push(sample_b);
    }

    (
        SampleSummary::from_samples(samples_a, per_sample_a)
            .expect("invariant: the non-zero sample count fills every timing sample"),
        SampleSummary::from_samples(samples_b, per_sample_b)
            .expect("invariant: the non-zero sample count fills every timing sample"),
    )
}

fn timed_block(operation: &mut impl FnMut(), iterations: u64) -> u128 {
    let sample_start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    normalized_picoseconds(sample_start.elapsed(), iterations)
}

/// Measures `operation` over inputs built outside the timed region.
///
/// [`measure`] times the whole closure, so a per-iteration reset — restoring
/// the input of an in-place operation — is charged to the operation. When two
/// arms reset buffers of different element widths, the difference between those
/// resets lands in the reported ratio between them, which is exactly the
/// quantity such a comparison exists to read. This prepares one input per
/// iteration first, times only the operations, and drops the inputs afterwards.
pub(crate) fn measure_batched<T>(
    config: BenchmarkConfig,
    mut setup: impl FnMut() -> T,
    mut operation: impl FnMut(&mut T),
) -> SampleSummary {
    if config.mode() == BenchmarkMode::Smoke {
        let mut input = setup();
        let sample_start = Instant::now();
        operation(&mut input);
        return SampleSummary::from_single_observation(normalized_picoseconds(
            sample_start.elapsed(),
            1,
        ));
    }

    // Warm-up carries its own setup, so calibration sees the whole cycle and
    // returns an iteration count whose inputs fit in memory.
    let warm_up_start = Instant::now();
    let mut warm_up_iterations = 0_u64;
    while warm_up_start.elapsed() < config.warm_up() {
        let mut input = setup();
        operation(&mut input);
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
        let mut inputs: Vec<T> = (0..iterations_per_sample).map(|_| setup()).collect();
        let sample_start = Instant::now();
        for input in &mut inputs {
            operation(input);
        }
        let elapsed = sample_start.elapsed();
        // Outside the reading: freeing the batch is not part of the operation.
        drop(inputs);
        samples.push(normalized_picoseconds(elapsed, iterations_per_sample));
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
