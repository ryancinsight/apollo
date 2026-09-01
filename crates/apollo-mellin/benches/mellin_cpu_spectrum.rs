//! Public CPU Mellin forward-spectrum execution across the Hermes threshold.
//!
//! The plan and analytical linear signal are constructed before timing. N = 64
//! stays below the provider threshold, while N = 128 and 256 exercise the
//! direct-DFT rows that borrow real samples and reuse retained complex weights.
//! An independent log-grid DFT validates one coefficient before each case.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use apollo_mellin::{MellinPlan, MellinSpectrum};
use eunomia::Complex64;
use std::hint::black_box;
use std::time::{Duration, Instant};

const GROUP: &str = "mellin_cpu_forward_spectrum";
const LENGTHS: [usize; 3] = [64, 128, 256];
const SIGNAL_MIN: f64 = 1.0;
const SIGNAL_MAX: f64 = 4.0;
const BUDGET_SECS: u64 = 30;

fn linear_signal(len: usize) -> Vec<f64> {
    let step = (SIGNAL_MAX - SIGNAL_MIN) / (len as f64 - 1.0);
    (0..len)
        .map(|index| 0.5 + 0.125 * (SIGNAL_MIN + index as f64 * step))
        .collect()
}

fn direct_log_frequency_bin(len: usize, bin: usize) -> Complex64 {
    let log_min = SIGNAL_MIN.ln();
    let log_max = SIGNAL_MAX.ln();
    let du = (log_max - log_min) / (len as f64 - 1.0);
    (0..len)
        .map(|index| {
            let u = log_min + index as f64 * du;
            let sample = 0.5 + 0.125 * u.exp();
            let angle = -std::f64::consts::TAU * bin as f64 * index as f64 / len as f64;
            Complex64::new(sample * angle.cos(), sample * angle.sin())
        })
        .sum::<Complex64>()
        * du
}

fn assert_matches_direct(spectrum: &MellinSpectrum, len: usize) {
    let bin = 7;
    let expected = direct_log_frequency_bin(len, bin);
    let actual = spectrum.values()[bin];
    let du = (SIGNAL_MAX.ln() - SIGNAL_MIN.ln()) / (len as f64 - 1.0);
    let l1 = (0..len)
        .map(|index| {
            let u = SIGNAL_MIN.ln() + index as f64 * du;
            (0.5 + 0.125 * u.exp()).abs()
        })
        .sum::<f64>()
        * du;
    // Linear interpolation is exact for this signal. The remaining direct-sum
    // and coordinate-evaluation error grows as O(N*u).
    let tolerance = 64.0 * len as f64 * f64::EPSILON * l1.max(1.0);
    assert!(
        (actual - expected).norm() <= tolerance,
        "N={len} bin={bin}: {actual:?} differs from {expected:?} by {}, bound {tolerance}",
        (actual - expected).norm()
    );
}

fn main() -> Result<(), apollo_bench::BenchmarkModeError> {
    let started = Instant::now();
    let mode = BenchmarkMode::from_environment()?;
    let mut suite = BenchmarkSuite::new(mode.apply(BenchmarkConfig::regression()));

    for len in LENGTHS {
        let plan = MellinPlan::new(len, SIGNAL_MIN, SIGNAL_MAX)
            .expect("benchmark Mellin dimensions are valid");
        let signal = linear_signal(len);
        let warm = plan
            .forward_spectrum(&signal, SIGNAL_MIN, SIGNAL_MAX)
            .expect("benchmark warm Mellin spectrum");
        assert_matches_direct(&warm, len);

        suite.run(
            BenchmarkCase::new(GROUP, format!("n_{len}"), "public_forward"),
            || {
                let spectrum = plan
                    .forward_spectrum(black_box(&signal), SIGNAL_MIN, SIGNAL_MAX)
                    .expect("benchmark Mellin spectrum");
                black_box(spectrum);
            },
        );
    }

    suite.emit();
    assert!(
        started.elapsed() < Duration::from_secs(BUDGET_SECS),
        "mellin_cpu_spectrum exceeded its {BUDGET_SECS}s budget ({:.2}s)",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
