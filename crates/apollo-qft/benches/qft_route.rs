//! Reusable QFT forward execution against the retained dense oracle.
//!
//! The plan, twiddles, input, and caller-owned outputs are constructed outside
//! the timed closures. Each length runs the public reusable plan and retained
//! dense/Hermes kernel in the same process against one immutable signal. The
//! boundary and geometrically spaced lengths cover identity, tiny transforms,
//! a non-smooth prime, and large quadratic workloads.
//!
//! The default budget is twenty-two cases times 20 ms warm-up plus 80 ms
//! measurement, or 2.2 s before harness overhead. A hard 30 s suite bound
//! catches an accidental return to unbounded quadratic work. Smoke mode still
//! executes and validates every case once.

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use apollo_qft::infrastructure::kernel::dense::qft_forward_dense_into;
use apollo_qft::{QftPlan, QuantumStateDimension};
use eunomia::Complex64;
use leto::Array1;
use std::hint::black_box;
use std::time::{Duration, Instant};

const GROUP: &str = "qft_forward_reusable";
const LENGTHS: [usize; 11] = [1, 2, 3, 4, 8, 16, 32, 64, 127, 256, 1024];
const WARM_UP_MS: u64 = 20;
const MEASUREMENT_MS: u64 = 80;
const BUDGET_SECS: u64 = 30;

fn signal(len: usize) -> Array1<Complex64> {
    Array1::from_shape_fn([len], |[index]| {
        let x = index as f64;
        Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
    })
}

fn twiddles(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / len as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect()
}

fn direct_qft(input: &Array1<Complex64>) -> Vec<Complex64> {
    let len = input.len();
    let scale = 1.0 / (len as f64).sqrt();
    (0..len)
        .map(|bin| {
            input
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let angle = std::f64::consts::TAU * (bin * index) as f64 / len as f64;
                    *value * Complex64::new(angle.cos(), angle.sin())
                })
                .sum::<Complex64>()
                * scale
        })
        .collect()
}

fn assert_matches_direct(input: &Array1<Complex64>, actual: &Array1<Complex64>) {
    let expected = direct_qft(input);
    let l1 = input.iter().map(|value| value.norm()).sum::<f64>();
    // Naive direct summation has O(N*u) forward error. The factor covers the
    // independently generated trigonometric values and the routed FFT's
    // O(log N*u) contribution without making the latter load-bearing.
    let tolerance = 64.0 * input.len() as f64 * f64::EPSILON * l1.max(1.0);
    for (bin, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*got - want).norm() <= tolerance,
            "QFT length {} bin {bin}: {got:?} differs from {want:?} by {}, bound {tolerance}",
            input.len(),
            (*got - want).norm()
        );
    }
}

fn main() -> Result<(), apollo_bench::BenchmarkModeError> {
    let started = Instant::now();
    let mode = BenchmarkMode::from_environment()?;
    let config = mode.apply(
        BenchmarkConfig::try_with_budgets(
            Duration::from_millis(WARM_UP_MS),
            Duration::from_millis(MEASUREMENT_MS),
        )
        .expect("invariant: QFT benchmark budgets are non-zero"),
    );
    let mut suite = BenchmarkSuite::new(config);

    for len in LENGTHS {
        let input = signal(len);
        let input_slice = input.as_slice().expect("benchmark input is contiguous");
        let twiddles = twiddles(len);
        let mut dense_output = vec![Complex64::new(0.0, 0.0); len];
        qft_forward_dense_into(input_slice, &mut dense_output, &twiddles);
        assert_matches_direct(&input, &Array1::from(dense_output.clone()));

        suite.run(BenchmarkCase::new(GROUP, "dense", len), || {
            qft_forward_dense_into(
                black_box(input_slice),
                black_box(&mut dense_output),
                black_box(&twiddles),
            );
            black_box(&dense_output);
        });

        let plan = QftPlan::new(
            QuantumStateDimension::new(len).expect("benchmark lengths are valid dimensions"),
        );
        let mut output = Array1::zeros([len]);
        plan.forward_into(&input, &mut output)
            .expect("benchmark QFT execution");
        assert_matches_direct(&input, &output);

        suite.run(BenchmarkCase::new(GROUP, "apollo", len), || {
            plan.forward_into(black_box(&input), black_box(&mut output))
                .expect("benchmark QFT execution");
            black_box(&output);
        });
    }

    suite.emit();
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(BUDGET_SECS),
        "qft_route exceeded its {BUDGET_SECS}s budget ({:.2}s)",
        elapsed.as_secs_f64()
    );
    Ok(())
}
