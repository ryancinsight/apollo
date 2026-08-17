//! Clone-inclusive 1D forward complex FFT, Apollo against RustFFT.
//!
//! This is the measurement half of `docs/benchmark_results.md`; `cargo run -p xtask
//! -- benchmark` runs this binary and renders the table from its CSV.
//!
//! ## What "clone-inclusive" means
//!
//! Both engines transform in place, so a timed iteration must restore its input
//! or it measures a different signal on every repeat. Each measured closure
//! therefore copies the pristine input into the working buffer and then
//! transforms it. That copy is charged to **both** engines identically, so it
//! cancels in the ratio while keeping each absolute figure honest about what a
//! caller pays to transform a buffer it still owns.
//!
//! Plans are built once, outside the timed region, for both engines. Planning
//! is a setup cost that a real caller amortises; including it would measure
//! planner construction rather than transform throughput.
//!
//! ## Runtime budget
//!
//! ```text
//! per case  = WARM_UP_MS + MEASUREMENT_MS         = 20 ms + 80 ms = 100 ms
//! cases     = 4 per size (Apollo/RustFFT x f64/f32)
//! default   = 22 sizes x 4 x 100 ms               ~ 8.8 s
//! full      = 500 sizes x 4 x 100 ms              ~ 200 s   (opt-in)
//! ```
//!
//! The default sweep is budgeted to sit inside the 30 s bound that every Apollo
//! bench binary observes. The full 1..=500 sweep is deliberately opt-in via
//! `APOLLO_BENCH_FULL_SWEEP=1`, because a dense linear grid is a long job that
//! belongs on a quiet machine, not in an ordinary gate.
//!
//! The bound is enforced rather than asserted in prose: [`main`] exits non-zero
//! if the default sweep exceeds [`BUDGET_SECS`]. A breach is a defect to
//! root-cause — an oversized workload here, or a genuinely slower kernel — and
//! is never fixed by raising the bound in the change that caused it.
//!
//! ## What this binary does not measure
//!
//! It reports timings only. `docs/benchmark_results.md` historically also carried an
//! engine-name column per row; neither engine exposes its selected algorithm
//! through a public API, so those columns cannot be regenerated and are not
//! invented here.

use std::time::{Duration, Instant};

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::{Complex32, Complex64};
use rustfft::num_complex::Complex as RustComplex;
use rustfft::FftPlanner;
use std::hint::black_box;

/// Hard wall-clock bound for the default sweep. See the module budget table.
const BUDGET_SECS: u64 = 30;
/// Per-case warm-up.
const WARM_UP_MS: u64 = 20;
/// Per-case measurement window.
const MEASUREMENT_MS: u64 = 80;
/// Largest length in the opt-in full sweep.
const FULL_SWEEP_MAX: usize = 500;
/// Environment switch selecting the dense 1..=FULL_SWEEP_MAX sweep.
const FULL_SWEEP_VAR: &str = "APOLLO_BENCH_FULL_SWEEP";

/// Representative lengths spanning the regimes Apollo dispatches differently:
/// identity, short Winograd codelets, power-of-two Stockham, odd primes that
/// reach Rader, prime-power and smooth composites, and a few larger sizes where
/// asymptotics rather than codelet quality dominate.
const DEFAULT_SIZES: [usize; 22] = [
    1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 19, 31, 32, 53, 64, 67, 96, 121, 128, 200, 256, 512,
];

/// Deterministic complex signal; identical values feed both engines.
fn signal(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

fn narrow(input: &[Complex64]) -> Vec<Complex32> {
    input
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect()
}

fn to_rustfft_64(input: &[Complex64]) -> Vec<RustComplex<f64>> {
    input
        .iter()
        .map(|value| RustComplex::new(value.re, value.im))
        .collect()
}

fn to_rustfft_32(input: &[Complex32]) -> Vec<RustComplex<f32>> {
    input
        .iter()
        .map(|value| RustComplex::new(value.re, value.im))
        .collect()
}

fn sizes() -> Vec<usize> {
    match std::env::var(FULL_SWEEP_VAR) {
        Ok(value) if value == "1" => (1..=FULL_SWEEP_MAX).collect(),
        _ => DEFAULT_SIZES.to_vec(),
    }
}

fn bench_size(suite: &mut BenchmarkSuite, config: BenchmarkConfig, len: usize) {
    const GROUP: &str = "fft_forward_clone_inclusive";

    let source64 = signal(len);
    let source32 = narrow(&source64);
    let rust_source64 = to_rustfft_64(&source64);
    let rust_source32 = to_rustfft_32(&source32);

    // Plans and working buffers are built once; only the copy plus transform is
    // timed. Both engines get the same treatment.
    let apollo64 = apollo_fft::FftPlan1D::<f64>::new(apollo_fft::Shape1D { n: len });
    let apollo32 = apollo_fft::FftPlan1D::<f32>::new(apollo_fft::Shape1D { n: len });
    let mut planner64 = FftPlanner::<f64>::new();
    let mut planner32 = FftPlanner::<f32>::new();
    let rust64 = planner64.plan_fft_forward(len);
    let rust32 = planner32.plan_fft_forward(len);

    let mut work64 = source64.clone();
    suite.run_with_config(config, BenchmarkCase::new(GROUP, "apollo_f64", len), || {
        work64.copy_from_slice(&source64);
        apollo64.forward_complex_slice_inplace(black_box(&mut work64));
        black_box(&work64);
    });

    let mut rust_work64 = rust_source64.clone();
    suite.run_with_config(
        config,
        BenchmarkCase::new(GROUP, "rustfft_f64", len),
        || {
            rust_work64.copy_from_slice(&rust_source64);
            rust64.process(black_box(&mut rust_work64));
            black_box(&rust_work64);
        },
    );

    let mut work32 = source32.clone();
    suite.run_with_config(config, BenchmarkCase::new(GROUP, "apollo_f32", len), || {
        work32.copy_from_slice(&source32);
        apollo32.forward_complex_slice_inplace(black_box(&mut work32));
        black_box(&work32);
    });

    let mut rust_work32 = rust_source32.clone();
    suite.run_with_config(
        config,
        BenchmarkCase::new(GROUP, "rustfft_f32", len),
        || {
            rust_work32.copy_from_slice(&rust_source32);
            rust32.process(black_box(&mut rust_work32));
            black_box(&rust_work32);
        },
    );
}

fn main() -> Result<(), apollo_bench::BenchmarkConfigError> {
    let started = Instant::now();
    let lengths = sizes();
    let full = lengths.len() > DEFAULT_SIZES.len();

    let config = BenchmarkConfig::try_with_budgets(
        Duration::from_millis(WARM_UP_MS),
        Duration::from_millis(MEASUREMENT_MS),
    )?;

    eprintln!(
        "rustfft_comparison: {} sweep, {} sizes, warm-up {WARM_UP_MS}ms, measurement {MEASUREMENT_MS}ms",
        if full { "full" } else { "default" },
        lengths.len()
    );
    if full {
        eprintln!(
            "rustfft_comparison: full sweep is opt-in and unbudgeted; run it on a quiet host"
        );
    } else {
        eprintln!("rustfft_comparison: budget {BUDGET_SECS}s (hard); set {FULL_SWEEP_VAR}=1 for 1..={FULL_SWEEP_MAX}");
    }

    let mut suite = BenchmarkSuite::new(config);
    for len in lengths {
        bench_size(&mut suite, config, len);
    }
    suite.emit();

    let elapsed = started.elapsed();
    eprintln!(
        "rustfft_comparison: completed in {:.2}s",
        elapsed.as_secs_f64()
    );

    // The full sweep is an explicit long job; only the default sweep is bounded.
    assert!(
        full || elapsed < Duration::from_secs(BUDGET_SECS),
        "rustfft_comparison default sweep exceeded its {BUDGET_SECS}s budget ({:.2}s). \
         Root-cause the slowdown; do not raise the bound in the change that caused it.",
        elapsed.as_secs_f64()
    );

    Ok(())
}
