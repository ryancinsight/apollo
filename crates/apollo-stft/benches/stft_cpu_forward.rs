//! Reusable CPU STFT forward execution with retained caller-owned output.
//!
//! The plan, Hann window, signal, and output are constructed before timing.
//! The three cases retain one scalar-windowing control and two complete-plan
//! workloads whose 1,024-point frames exercise the provider-backed windowing
//! regime. An independent direct DFT validates one unpadded frame before each
//! case enters the timed closure.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use apollo_stft::StftPlan;
use eunomia::Complex64;
use leto::Array1;
use std::hint::black_box;
use std::time::{Duration, Instant};

const GROUP: &str = "stft_cpu_forward_reuse";
const PARAMETERS: [(usize, usize, usize); 3] =
    [(32, 16, 4_096), (1_024, 512, 16_384), (1_024, 512, 65_536)];
const BUDGET_SECS: u64 = 30;

fn analytical_signal(signal_len: usize, frame_len: usize) -> Array1<f64> {
    Array1::from(
        (0..signal_len)
            .map(|index| {
                let phase = std::f64::consts::TAU * index as f64 / frame_len as f64;
                (7.0 * phase).sin() + 0.375 * (11.0 * phase).cos()
            })
            .collect::<Vec<_>>(),
    )
}

fn direct_frame_bin(signal: &[f64], window: &[f64], bin: usize) -> Complex64 {
    let frame_len = window.len();
    signal.iter().zip(window).take(frame_len).enumerate().fold(
        Complex64::new(0.0, 0.0),
        |sum, (index, (&sample, &factor))| {
            let angle = -std::f64::consts::TAU * bin as f64 * index as f64 / frame_len as f64;
            sum + Complex64::new(sample * factor * angle.cos(), sample * factor * angle.sin())
        },
    )
}

fn assert_matches_direct(plan: &StftPlan, signal: &Array1<f64>, output: &Array1<Complex64>) {
    let frame_len = plan.frame_len();
    let bin = 7.min(frame_len - 1);
    // Frame one begins at signal index zero because every benchmark uses the
    // COLA half-frame hop. This avoids boundary padding in the independent DFT.
    let expected = direct_frame_bin(
        signal.as_slice().expect("benchmark signal is contiguous"),
        plan.window()
            .as_slice()
            .expect("benchmark window is contiguous"),
        bin,
    );
    let actual = output[frame_len + bin];
    let l1 = signal
        .iter()
        .take(frame_len)
        .zip(plan.window())
        .map(|(&sample, &factor)| (sample * factor).abs())
        .sum::<f64>();
    // The direct sum contributes O(N*u) error; the routed FFT contributes
    // O(log(N)*u). The factor also covers independently evaluated twiddles.
    let tolerance = 64.0 * frame_len as f64 * f64::EPSILON * l1.max(1.0);
    assert!(
        (actual - expected).norm() <= tolerance,
        "frame {frame_len} bin {bin}: {actual:?} differs from {expected:?} by {}, bound {tolerance}",
        (actual - expected).norm()
    );
}

fn main() -> Result<(), apollo_bench::BenchmarkModeError> {
    let started = Instant::now();
    let mode = BenchmarkMode::from_environment()?;
    let mut suite = BenchmarkSuite::new(mode.apply(BenchmarkConfig::regression()));

    for (frame_len, hop_len, signal_len) in PARAMETERS {
        let plan = StftPlan::new(frame_len, hop_len).expect("benchmark dimensions are valid");
        let signal = analytical_signal(signal_len, frame_len);
        let mut output =
            Array1::<Complex64>::zeros([plan.frame_count(signal_len) * plan.spectrum_len()]);
        plan.forward_into(&signal, &mut output)
            .expect("benchmark warm forward STFT");
        assert_matches_direct(&plan, &signal, &output);

        suite.run(
            BenchmarkCase::new(
                GROUP,
                format!("frame_{frame_len}"),
                format!("signal_{signal_len}"),
            ),
            || {
                plan.forward_into(black_box(&signal), black_box(&mut output))
                    .expect("benchmark forward STFT");
                black_box(&output);
            },
        );
    }

    suite.emit();
    assert!(
        started.elapsed() < Duration::from_secs(BUDGET_SECS),
        "stft_cpu_forward exceeded its {BUDGET_SECS}s budget ({:.2}s)",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
