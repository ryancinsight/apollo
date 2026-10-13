//! Typed DWT forward/inverse timing: one flat, finest-first detail buffer
//! against the per-level heap row it replaces (ATLAS-ARCH-008).
//!
//! Run with `cargo bench -p apollo-wavelet --bench dwt_typed_flat_details`.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use apollo_fft::PrecisionProfile;
use apollo_wavelet::{DiscreteWavelet, DwtPlan};
use std::hint::black_box;

fn signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = index as f32;
            (0.017 * value).sin() + 0.25 * (0.31 * value).cos()
        })
        .collect()
}

fn bench_typed_round_trip(suite: &mut BenchmarkSuite, config: BenchmarkConfig) {
    for (len, levels) in [(4096_usize, 6_usize), (16384, 8)] {
        let plan = DwtPlan::new(len, levels, DiscreteWavelet::Haar).expect("valid DWT plan");
        let signal = signal(len);
        let mut approximation = vec![0.0_f32; len >> levels];
        let mut details = vec![0.0_f32; len - (len >> levels)];
        let mut recovered = vec![0.0_f32; len];

        suite.run_with_config(
            config,
            BenchmarkCase::new("dwt_typed_round_trip", "forward_inverse_f32", len),
            || {
                plan.forward_typed_into(
                    black_box(&signal),
                    &mut approximation,
                    &mut details,
                    PrecisionProfile::LOW_PRECISION_F32,
                )
                .expect("forward");
                plan.inverse_typed_into(
                    &approximation,
                    &details,
                    &mut recovered,
                    PrecisionProfile::LOW_PRECISION_F32,
                )
                .expect("inverse");
                black_box(&recovered);
            },
        );
    }
}

fn main() -> Result<(), apollo_bench::BenchmarkModeError> {
    let mode = BenchmarkMode::from_environment()?;
    let config = mode.apply(BenchmarkConfig::regression());
    let mut suite = BenchmarkSuite::new(config);
    bench_typed_round_trip(&mut suite, config);
    print!("{}", suite.report());
    Ok(())
}
