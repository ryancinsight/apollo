//! Clone-inclusive 1D forward complex FFT on powers of two, Apollo against PhastFT.
//!
//! ## Why this is a separate binary from `rustfft_comparison`
//!
//! `rustfft_comparison` sweeps arbitrary lengths and feeds the table in
//! `docs/benchmark_results.md`. PhastFT is power-of-two only — `PlannerDit64::new`
//! panics otherwise — so two thirds of that sweep's rows would be empty for it,
//! and its CSV schema is consumed by `xtask`. This binary measures a different
//! scenario rather than a second engine in the same one: the power-of-two
//! domain, on the split real/imaginary layout PhastFT is designed around.
//!
//! ## What is timed, and what is charged to whom
//!
//! Both engines transform in place, so a timed iteration must restore its input
//! or it measures a different signal on every repeat. Each measured closure
//! copies the pristine input into the working buffer and then transforms it.
//! The copy moves the same number of bytes for both engines — one interleaved
//! buffer for Apollo, two planes for PhastFT — so it cancels in the ratio.
//!
//! Plans are built once, outside the timed region, for both engines
//! (`FftPlan1D::new` and `PlannerDit64::new`). Planning is a setup cost a real
//! caller amortises.
//!
//! ## Layout, stated rather than hidden
//!
//! PhastFT is measured through its native split real/imaginary entry
//! (`fft_f64_dit_with_planner`), which is its best case. A caller holding
//! interleaved complex data pays either a deinterleave or PhastFT's own
//! `fft_*_dit_interleaved` path; neither is measured here. Apollo is measured on
//! the interleaved buffer its public slice API takes. The ratio therefore
//! compares each engine on the layout it is built for, not two engines on one
//! layout — read it as such.
//!
//! ## Runtime budget
//!
//! ```text
//! per case  = WARM_UP_MS + MEASUREMENT_MS   = 20 ms + 80 ms = 100 ms
//! cases     = 4 per size (Apollo/PhastFT x f64/f32)
//! total     = 8 sizes x 4 x 100 ms          ~ 3.2 s + at most one iteration
//!                                             of overshoot per case
//! ```
//!
//! Sizes are geometric across the cache regimes — L1-resident through
//! well past last-level cache — rather than a dense grid, because the question
//! this instrument answers is where the two engines diverge by regime.
//!
//! [`main`] exits non-zero if the sweep exceeds [`BUDGET_SECS`]. A breach is a
//! defect to root-cause — an oversized workload here, or a genuinely slower
//! kernel — and is never fixed by raising the bound in the change that caused it.

use std::hint::black_box;
use std::time::{Duration, Instant};

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use eunomia::{Complex32, Complex64};
use phastft::planner::{Direction, PlannerDit32, PlannerDit64};
use phastft::{fft_f32_dit_with_planner, fft_f64_dit_with_planner};

/// Hard wall-clock bound for the sweep. See the module budget table.
const BUDGET_SECS: u64 = 30;
/// Per-case warm-up.
const WARM_UP_MS: u64 = 20;
/// Per-case measurement window.
const MEASUREMENT_MS: u64 = 80;

/// Powers of two spanning the regimes that separate these two designs: a
/// working set inside L1, inside L2, inside last-level cache, and past it,
/// where PhastFT's in-place formulation and cache-optimal bit reversal are
/// meant to pay and Apollo's Stockham ping-pong pays its second buffer.
///
/// One representative per decade rather than a dense grid: a dense sweep costs
/// wall-clock without adding a regime.
const SIZES: [usize; 8] = [64, 256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];

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

/// Split an interleaved signal into the real and imaginary planes PhastFT takes.
fn split_64(input: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
    (
        input.iter().map(|value| value.re).collect(),
        input.iter().map(|value| value.im).collect(),
    )
}

fn split_32(input: &[Complex32]) -> (Vec<f32>, Vec<f32>) {
    (
        input.iter().map(|value| value.re).collect(),
        input.iter().map(|value| value.im).collect(),
    )
}

fn bench_size(suite: &mut BenchmarkSuite, config: BenchmarkConfig, len: usize) {
    const GROUP: &str = "fft_forward_power_of_two";

    let source64 = signal(len);
    let source32 = narrow(&source64);
    let (re64, im64) = split_64(&source64);
    let (re32, im32) = split_32(&source32);

    let apollo64 = apollo_fft::FftPlan1D::<f64>::new(
        apollo_fft::Shape1D::new(len).expect("invariant: shape lengths are non-zero"),
    );
    let apollo32 = apollo_fft::FftPlan1D::<f32>::new(
        apollo_fft::Shape1D::new(len).expect("invariant: shape lengths are non-zero"),
    );
    let phast64 = PlannerDit64::new(len);
    let phast32 = PlannerDit32::new(len);

    let mut work64 = source64.clone();
    suite.run_with_config(config, BenchmarkCase::new(GROUP, "apollo_f64", len), || {
        work64.copy_from_slice(&source64);
        apollo64.forward_complex_slice_inplace(black_box(&mut work64));
        black_box(&work64);
    });

    let mut phast_re64 = re64.clone();
    let mut phast_im64 = im64.clone();
    suite.run_with_config(
        config,
        BenchmarkCase::new(GROUP, "phastft_f64", len),
        || {
            phast_re64.copy_from_slice(&re64);
            phast_im64.copy_from_slice(&im64);
            fft_f64_dit_with_planner(
                black_box(&mut phast_re64),
                black_box(&mut phast_im64),
                Direction::Forward,
                &phast64,
            );
            black_box(&phast_re64);
        },
    );

    let mut work32 = source32.clone();
    suite.run_with_config(config, BenchmarkCase::new(GROUP, "apollo_f32", len), || {
        work32.copy_from_slice(&source32);
        apollo32.forward_complex_slice_inplace(black_box(&mut work32));
        black_box(&work32);
    });

    let mut phast_re32 = re32.clone();
    let mut phast_im32 = im32.clone();
    suite.run_with_config(
        config,
        BenchmarkCase::new(GROUP, "phastft_f32", len),
        || {
            phast_re32.copy_from_slice(&re32);
            phast_im32.copy_from_slice(&im32);
            fft_f32_dit_with_planner(
                black_box(&mut phast_re32),
                black_box(&mut phast_im32),
                Direction::Forward,
                &phast32,
            );
            black_box(&phast_re32);
        },
    );
}

fn main() -> Result<(), apollo_bench::BenchmarkModeError> {
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
        "phastft_comparison: {mode:?} mode, {} sizes, measurement configuration warm-up {WARM_UP_MS}ms and measurement {MEASUREMENT_MS}ms, budget {BUDGET_SECS}s (hard)",
        SIZES.len()
    );

    let mut suite = BenchmarkSuite::new(config);
    for len in SIZES {
        bench_size(&mut suite, config, len);
    }
    print!("{}", suite.report());

    let elapsed = started.elapsed();
    eprintln!(
        "phastft_comparison: completed in {:.2}s",
        elapsed.as_secs_f64()
    );

    assert!(
        elapsed < Duration::from_secs(BUDGET_SECS),
        "phastft_comparison exceeded its {BUDGET_SECS}s budget ({:.2}s). \
         Root-cause the slowdown; do not raise the bound in the change that caused it.",
        elapsed.as_secs_f64()
    );

    Ok(())
}
