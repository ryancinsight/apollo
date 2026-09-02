//! Native Apollo benchmarks for the CWT scale row: direct per-coefficient
//! evaluation against the FFT cross-correlation.
//!
//! Run with `cargo bench -p apollo-wavelet --bench cwt_scale_rows`.
//!
//! # Instrument design
//!
//! The harness takes 100 samples per case, so a case costs about
//! `max(measurement_budget, 100 x iteration)`. The direct row is
//! `O(n^2)` mother-wavelet evaluations, which puts `n = 4096` at seconds per
//! iteration — a workload, not an operation. The paired A/B therefore stops at
//! `n = 1024`, where the direct row still fits the budget, and the larger
//! sizes measure the convolution row alone to show its `O(n log n)` scaling.
//! The machine-independent evidence is the evaluation count, which the module
//! documentation of `continuous::convolution` derives exactly.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use apollo_wavelet::infrastructure::kernel::continuous::coefficient;
use apollo_wavelet::infrastructure::kernel::continuous::convolution::CwtSpectrum;
use apollo_wavelet::ContinuousWavelet;
use std::hint::black_box;

const WAVELET: ContinuousWavelet = ContinuousWavelet::Morlet { omega0: 5.0 };
const SCALE: f64 = 8.0;

fn signal(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let value = index as f64;
            (0.017 * value).sin() + 0.25 * (0.31 * value).cos()
        })
        .collect()
}

/// Paired direct-against-convolution rows at the sizes where both fit the
/// per-case budget.
fn bench_paired_scale_row(suite: &mut BenchmarkSuite, config: BenchmarkConfig) {
    // Geometric points straddling `FFT_CWT_LEN_THRESHOLD`, extended down to
    // the sizes where the transform's fixed costs can still dominate.
    for len in [4_usize, 8, 16, 32, 64, 256, 1024] {
        let samples = signal(len);
        let mut row = vec![0.0; len];

        suite.run_with_config(
            config,
            BenchmarkCase::new("cwt_scale_row", "direct", len),
            || {
                for (shift, slot) in row.iter_mut().enumerate() {
                    *slot = coefficient(black_box(&samples), WAVELET, SCALE, shift);
                }
                black_box(&row);
            },
        );

        let spectrum = CwtSpectrum::new(&samples);
        let mut fft_row = vec![0.0; len];
        suite.run_with_config(
            config,
            BenchmarkCase::new("cwt_scale_row", "fft_convolution", len),
            || {
                black_box(&spectrum).row_into(WAVELET, SCALE, &mut fft_row);
                black_box(&fft_row);
            },
        );

        // Worst case for the convolution path: a one-scale transform, where
        // the signal spectrum is built and then used exactly once instead of
        // being amortized across the scale rows.
        suite.run_with_config(
            config,
            BenchmarkCase::new("cwt_scale_row", "fft_convolution_single_scale", len),
            || {
                let spectrum = CwtSpectrum::new(black_box(&samples));
                spectrum.row_into(WAVELET, SCALE, &mut fft_row);
                black_box(&fft_row);
            },
        );
    }
}

/// Convolution-row scaling past the size where the direct row is measurable.
fn bench_convolution_scaling(suite: &mut BenchmarkSuite, config: BenchmarkConfig) {
    for len in [4096_usize, 16384] {
        let samples = signal(len);
        let spectrum = CwtSpectrum::new(&samples);
        let mut row = vec![0.0; len];
        suite.run_with_config(
            config,
            BenchmarkCase::new("cwt_scale_row", "fft_convolution", len),
            || {
                black_box(&spectrum).row_into(WAVELET, SCALE, &mut row);
                black_box(&row);
            },
        );
    }
}

fn main() -> Result<(), apollo_bench::BenchmarkModeError> {
    let mode = BenchmarkMode::from_environment()?;
    let config = mode.apply(BenchmarkConfig::regression());
    let mut suite = BenchmarkSuite::new(config);
    bench_paired_scale_row(&mut suite, config);
    bench_convolution_scaling(&mut suite, config);
    print!("{}", suite.report());
    Ok(())
}
