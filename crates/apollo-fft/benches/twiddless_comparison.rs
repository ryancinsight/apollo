//! Clone-inclusive 1D forward complex FFT: the twiddless algorithm, its two
//! butterfly isomorphs, Apollo's production plan and RustFFT.
//!
//! ## Hypothesis and stop condition
//!
//! Queiroz's twiddless FFT (arXiv:2505.23718v2) claims structural advantages
//! over butterfly FFTs: no butterfly, a combination step that is only index
//! reordering, and lengths `c · 2^k` without zero padding. The kernel under
//! test executes Algorithm 1 as published (`CompressedHalves`) beside two
//! schedules that perform bit-identical arithmetic in place
//! (`ButterflyRecursive`, `ButterflyIterative`), so the arms differ in data
//! movement only. The prediction, from the paper's own count of `7N·log₂N`
//! real operations against `5N·log₂N`, is that the published schedule is no
//! faster than its in-place isomorph at any length and slower than Apollo's
//! radix-4/8 Stockham and codelet routes at every power of two. The
//! measurement decides; the comparator's family-wise disjoint intervals are the
//! acceptance criterion, per `apollo-bench`.
//!
//! ## What "clone-inclusive" means
//!
//! Apollo and RustFFT transform in place, so a timed iteration restores its
//! input first. The experimental arms transform out of place and would not
//! need the copy; it is charged to all five arms identically so that the copy
//! cancels in every ratio. Plans, twiddle tables, output buffers and scratch
//! are built once outside the timed region for every arm.
//!
//! ## Runtime budget
//!
//! ```text
//! per case  = WARM_UP_MS + MEASUREMENT_MS         = 20 ms + 80 ms = 100 ms
//! cases     = 5 arms x 2 precisions               = 10 per size
//! default   = 11 sizes x 10 x 100 ms              ~ 11.0 s
//! ```
//!
//! The lengths are geometric across the power-of-two regimes Apollo routes
//! differently (codelet, Stockham, split base, four-step) plus the
//! `3 · 2^k` and `5 · 2^k` lengths the paper singles out, including its own
//! example `5120`. [`main`] exits non-zero if the sweep exceeds
//! [`BUDGET_SECS`]; a breach is root-caused, never absorbed by the bound.

use std::time::{Duration, Instant};

use apollo_bench::{
    bind_measurement_processor, BenchmarkCase, BenchmarkConfig, BenchmarkError, BenchmarkMode,
    BenchmarkSuite,
};
use apollo_fft::application::execution::kernel::twiddless::{
    ButterflyIterative, ButterflyRecursive, CompressedHalves, Schedule, TwiddlessPlan,
    TwiddlessScalar,
};
use apollo_fft::{FftPlan1D, Shape1D};
use eunomia::{Complex, FloatElement};
use rustfft::num_complex::Complex as RustComplex;
use rustfft::{FftNum, FftPlanner};
use std::hint::black_box;

/// Hard wall-clock bound for the sweep. See the module budget table.
const BUDGET_SECS: u64 = 30;
/// Per-case warm-up.
const WARM_UP_MS: u64 = 20;
/// Per-case measurement window.
const MEASUREMENT_MS: u64 = 80;
/// Measured lengths: powers of two across Apollo's routing regimes, then the
/// paper's `c · 2^k` lengths.
const SIZES: [usize; 11] = [
    16, 64, 256, 1_024, 4_096, 16_384, 65_536, 96, 640, 5_120, 12_288,
];
const GROUP: &str = "fft_forward_clone_inclusive";

/// Deterministic complex signal; identical values feed every arm.
fn signal<F: FloatElement>(len: usize) -> Vec<Complex<F>> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex::new(
                F::from_f64((0.017 * x).sin()),
                F::from_f64(0.25 * (0.031 * x).cos()),
            )
        })
        .collect()
}

fn to_rustfft<F: FloatElement>(input: &[Complex<F>]) -> Vec<RustComplex<F>> {
    input
        .iter()
        .map(|value| RustComplex::new(value.re, value.im))
        .collect()
}

fn bench_schedule<F, S>(
    suite: &mut BenchmarkSuite,
    config: BenchmarkConfig,
    operation: &str,
    plan: &TwiddlessPlan<F>,
    source: &[Complex<F>],
) where
    F: TwiddlessScalar,
    S: Schedule,
{
    let len = source.len();
    let mut work = source.to_vec();
    let mut output = vec![Complex::<F>::default(); len];
    let mut scratch = vec![Complex::<F>::default(); plan.scratch_len()];
    suite.run_with_config(config, BenchmarkCase::new(GROUP, operation, len), || {
        work.copy_from_slice(source);
        plan.forward::<S>(
            black_box(&work),
            black_box(&mut output),
            black_box(&mut scratch),
        );
        black_box(&output);
    });
}

/// One precision at one length: the three schedules, Apollo's plan through
/// `production` (built by the caller outside the timed region), and RustFFT.
fn bench_precision<F, P>(
    suite: &mut BenchmarkSuite,
    config: BenchmarkConfig,
    label: &str,
    len: usize,
    production: P,
) where
    F: TwiddlessScalar + FloatElement + FftNum,
    P: Fn(&mut [Complex<F>]),
{
    let source = signal::<F>(len);
    let plan = TwiddlessPlan::<F>::new(len).expect("invariant: SIZES are admitted lengths");

    bench_schedule::<F, CompressedHalves>(
        suite,
        config,
        &format!("twiddless_{label}"),
        &plan,
        &source,
    );
    bench_schedule::<F, ButterflyRecursive>(
        suite,
        config,
        &format!("butterfly_recursive_{label}"),
        &plan,
        &source,
    );
    bench_schedule::<F, ButterflyIterative>(
        suite,
        config,
        &format!("butterfly_iterative_{label}"),
        &plan,
        &source,
    );

    let mut work = source.clone();
    suite.run_with_config(
        config,
        BenchmarkCase::new(GROUP, format!("apollo_{label}"), len),
        || {
            work.copy_from_slice(&source);
            production(black_box(&mut work));
            black_box(&work);
        },
    );

    let rust_source = to_rustfft(&source);
    let rust = FftPlanner::<F>::new().plan_fft_forward(len);
    let mut rust_scratch = vec![RustComplex::new(F::ZERO, F::ZERO); rust.get_inplace_scratch_len()];
    let mut rust_work = rust_source.clone();
    suite.run_with_config(
        config,
        BenchmarkCase::new(GROUP, format!("rustfft_{label}"), len),
        || {
            rust_work.copy_from_slice(&rust_source);
            rust.process_with_scratch(black_box(&mut rust_work), black_box(&mut rust_scratch));
            black_box(&rust_work);
        },
    );
}

fn main() -> Result<(), BenchmarkError> {
    let processor = bind_measurement_processor()?;
    eprintln!("twiddless_comparison: {}", processor.describe());
    let started = Instant::now();
    let mode = BenchmarkMode::from_environment()?;
    let config = mode.apply(
        BenchmarkConfig::try_with_budgets(
            Duration::from_millis(WARM_UP_MS),
            Duration::from_millis(MEASUREMENT_MS),
        )
        .expect("invariant: benchmark duration constants are non-zero"),
    );
    eprintln!(
        "twiddless_comparison: {mode:?} mode, {} sizes, warm-up {WARM_UP_MS}ms and measurement {MEASUREMENT_MS}ms per case, budget {BUDGET_SECS}s (hard)",
        SIZES.len()
    );

    let mut suite = BenchmarkSuite::new(config);
    for len in SIZES {
        let shape = Shape1D::new(len).expect("invariant: SIZES are non-zero");
        let apollo64 = FftPlan1D::<f64>::new(shape);
        bench_precision::<f64, _>(&mut suite, config, "f64", len, |work| {
            apollo64.forward_complex_slice_inplace(work);
        });
        let apollo32 = FftPlan1D::<f32>::new(shape);
        bench_precision::<f32, _>(&mut suite, config, "f32", len, |work| {
            apollo32.forward_complex_slice_inplace(work);
        });
    }
    print!("{}", suite.report());

    let elapsed = started.elapsed();
    eprintln!(
        "twiddless_comparison: completed in {:.2}s",
        elapsed.as_secs_f64()
    );
    assert!(
        elapsed < Duration::from_secs(BUDGET_SECS),
        "twiddless_comparison exceeded its {BUDGET_SECS}s budget ({:.2}s). \
         Root-cause the slowdown; do not raise the bound in the change that caused it.",
        elapsed.as_secs_f64()
    );
    Ok(())
}
