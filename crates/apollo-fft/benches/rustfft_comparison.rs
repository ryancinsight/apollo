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
//! Plans and reusable scratch are built once, outside the timed region, for
//! both engines. Planning and scratch allocation are setup costs that a real
//! caller amortises; including either would measure lifecycle overhead rather
//! than transform throughput. In particular, RustFFT's convenience `process`
//! method allocates and zero-fills scratch on every call, so this instrument
//! uses its allocation-free `process_with_scratch` execution seam.
//!
//! ## Runtime budget
//!
//! Each size runs two interleaved pairs (Apollo/RustFFT at f64, then at
//! f32): sequential per-engine windows alias with package-turbo
//! oscillation (#639), so both arms of a pair share one warm-up and
//! alternate timed sample blocks within one window.
//!
//! ```text
//! per pair  = WARM_UP_MS + 2 x MEASUREMENT_MS     = 20 ms + 160 ms = 180 ms (base)
//!             sizes at and above 1024 scale each arm's measurement window
//!             by len / 512, so a 1024-point pair gets 20 + 2 x 160 ms,
//!             stabilizing the 100-sample medians against scheduler noise.
//! pairs     = 2 per size (f64, f32)
//! default   = 26 sizes x 2 x ~200 ms (avg)        ~ 11 s
//! full      = 500 sizes x 2 x 180 ms              ~ 180 s   (opt-in)
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

use apollo_bench::{
    bind_measurement_processor, BenchmarkCase, BenchmarkConfig, BenchmarkError, BenchmarkMode,
    BenchmarkSuite, LaneBuffer,
};
use eunomia::{Complex32, Complex64};
use rustfft::num_complex::Complex as RustComplex;
use rustfft::FftPlanner;
use std::hint::black_box;

/// Hard wall-clock bound for the default sweep. See the module budget table.
const BUDGET_SECS: u64 = 30;
/// Per-case warm-up.
const WARM_UP_MS: u64 = 20;
/// Per-case base measurement window, scaled for sizes at and above
/// [`LARGE_SIZE_THRESHOLD`] so the 100-sample median collects enough
/// iterations to resist scheduler noise.
const MEASUREMENT_MS: u64 = 80;
/// Sizes at and above this threshold get a scaled measurement budget.
/// Below it the flat 80 ms window admits enough iterations per sample
/// for a stable median; at and above it the per-iteration cost is large
/// enough that 80 ms / 100 samples leaves each sample with too few
/// iterations, and the median swings 3x between runs.
const LARGE_SIZE_THRESHOLD: usize = 1024;
/// The base size for scaling: a transform of this length gets the base
/// measurement window, and larger lengths get a proportionally larger
/// window. Set below [`LARGE_SIZE_THRESHOLD`] so the threshold itself
/// gets a 2x budget rather than 1x.
const SCALE_BASE: usize = 512;
/// Largest length in the opt-in full sweep.
const FULL_SWEEP_MAX: usize = 500;
/// Environment switch selecting the dense 1..=FULL_SWEEP_MAX sweep.
const FULL_SWEEP_VAR: &str = "APOLLO_BENCH_FULL_SWEEP";
/// In-page lanes for the measurement roles (#646). Distinct mod 4096 with
/// L1-set spread (sets 0/32/16/48/8); the copy pair sits on opposite page
/// halves. The f32 pair reuses these lanes in its disjoint window.
const LANE_SOURCE: usize = 0;
const LANE_WORK: usize = 2048;
const LANE_RUST_SOURCE: usize = 1024;
const LANE_RUST_WORK: usize = 3072;
const LANE_SCRATCH: usize = 512;
/// Representative lengths spanning the regimes Apollo dispatches differently:
/// identity, short Winograd codelets, power-of-two Stockham, odd primes that
/// reach Rader, prime-power and smooth composites, and the sizes where the
/// per-length specializations at 1024, 2048, 4096, and 8192 live — which a
/// sweep stopping at 512 could not see.
const DEFAULT_SIZES: [usize; 26] = [
    1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 19, 31, 32, 53, 64, 67, 96, 121, 128, 200, 256, 512, 1024,
    2048, 4096, 8192,
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

/// Returns the measurement configuration for `len`, scaling the base budget
/// at and above [`LARGE_SIZE_THRESHOLD`] so the 100-sample median collects
/// enough iterations to resist scheduler noise. The mode (smoke vs measurement)
/// is preserved from `base`.
fn config_for(len: usize, base: BenchmarkConfig, mode: BenchmarkMode) -> BenchmarkConfig {
    if len < LARGE_SIZE_THRESHOLD {
        return base;
    }
    // Scale the measurement window by the transform-length ratio above the
    // scale base, so a 1024-point case gets 2x the base (160 ms), a 2048
    // gets 4x (320 ms), etc. This keeps iterations per sample roughly
    // constant across sizes rather than inversely proportional to length.
    let scale = len / SCALE_BASE;
    let scaled_measurement = MEASUREMENT_MS.saturating_mul(scale as u64);
    mode.apply(
        BenchmarkConfig::try_with_budgets(
            Duration::from_millis(WARM_UP_MS),
            Duration::from_millis(scaled_measurement),
        )
        .expect("invariant: scaled budgets are non-zero"),
    )
}

fn bench_size(
    suite: &mut BenchmarkSuite,
    base_config: BenchmarkConfig,
    mode: BenchmarkMode,
    len: usize,
) {
    const GROUP: &str = "fft_forward_clone_inclusive";
    let config = config_for(len, base_config, mode);

    let source64 = LaneBuffer::from_slice(&signal(len), LANE_SOURCE);
    let source32 = LaneBuffer::from_slice(&narrow(source64.as_slice()), LANE_SOURCE);
    let rust_source64 =
        LaneBuffer::from_slice(&to_rustfft_64(source64.as_slice()), LANE_RUST_SOURCE);
    let rust_source32 =
        LaneBuffer::from_slice(&to_rustfft_32(source32.as_slice()), LANE_RUST_SOURCE);

    // Plans, working buffers, and reusable scratch are built once; only the
    // copy plus transform is timed. Both engines get the same treatment.
    let apollo64 = apollo_fft::FftPlan1D::<f64>::new(
        apollo_fft::Shape1D::new(len).expect("invariant: shape lengths are non-zero"),
    );
    let apollo32 = apollo_fft::FftPlan1D::<f32>::new(
        apollo_fft::Shape1D::new(len).expect("invariant: shape lengths are non-zero"),
    );
    let mut planner64 = FftPlanner::<f64>::new();
    let mut planner32 = FftPlanner::<f32>::new();
    let rust64 = planner64.plan_fft_forward(len);
    let rust32 = planner32.plan_fft_forward(len);
    let mut rust_scratch64 = LaneBuffer::zeros(rust64.get_inplace_scratch_len(), LANE_SCRATCH);
    let mut rust_scratch32 = LaneBuffer::zeros(rust32.get_inplace_scratch_len(), LANE_SCRATCH);

    // Paired per precision: sequential per-engine windows alias with
    // package-turbo oscillation (#639), so each pair shares one window
    // and the medians are directly comparable.
    // Deterministic in-page lanes (#646): plain heap buffers slide with
    // ASLR and the low address bits decide 4 KiB-aliasing stalls, which
    // swung N=32 medians +-70-100% across runs of one binary. Distinct
    // lanes per live role (opposite page halves for the copy pair, L1-set
    // spread otherwise) pin the aliasing configuration so ratios compare
    // across runs; the f32 pair reuses lanes in its disjoint window.
    let mut work64 = LaneBuffer::from_slice(source64.as_slice(), LANE_WORK);
    let mut rust_work64 = LaneBuffer::from_slice(rust_source64.as_slice(), LANE_RUST_WORK);
    suite.run_paired_with_config(
        config,
        BenchmarkCase::new(GROUP, "apollo_f64", len),
        || {
            work64.as_mut_slice().copy_from_slice(source64.as_slice());
            apollo64.forward_complex_slice_inplace(black_box(work64.as_mut_slice()));
            black_box(work64.as_slice());
        },
        BenchmarkCase::new(GROUP, "rustfft_f64", len),
        || {
            rust_work64
                .as_mut_slice()
                .copy_from_slice(rust_source64.as_slice());
            rust64.process_with_scratch(
                black_box(rust_work64.as_mut_slice()),
                black_box(rust_scratch64.as_mut_slice()),
            );
            black_box(rust_work64.as_slice());
        },
    );

    let mut work32 = LaneBuffer::from_slice(source32.as_slice(), LANE_WORK);
    let mut rust_work32 = LaneBuffer::from_slice(rust_source32.as_slice(), LANE_RUST_WORK);
    suite.run_paired_with_config(
        config,
        BenchmarkCase::new(GROUP, "apollo_f32", len),
        || {
            work32.as_mut_slice().copy_from_slice(source32.as_slice());
            apollo32.forward_complex_slice_inplace(black_box(work32.as_mut_slice()));
            black_box(work32.as_slice());
        },
        BenchmarkCase::new(GROUP, "rustfft_f32", len),
        || {
            rust_work32
                .as_mut_slice()
                .copy_from_slice(rust_source32.as_slice());
            rust32.process_with_scratch(
                black_box(rust_work32.as_mut_slice()),
                black_box(rust_scratch32.as_mut_slice()),
            );
            black_box(rust_work32.as_slice());
        },
    );
}

fn main() -> Result<(), BenchmarkError> {
    let processor = bind_measurement_processor()?;
    eprintln!("rustfft_comparison: {}", processor.describe());
    let started = Instant::now();
    let lengths = sizes();
    let full = lengths.len() > DEFAULT_SIZES.len();
    let mode = BenchmarkMode::from_environment()?;

    let config = mode.apply(
        BenchmarkConfig::try_with_budgets(
            Duration::from_millis(WARM_UP_MS),
            Duration::from_millis(MEASUREMENT_MS),
        )
        .expect("invariant: benchmark duration constants are non-zero"),
    );

    eprintln!(
        "rustfft_comparison: {mode:?} mode, {} sweep, {} sizes, base measurement warm-up {WARM_UP_MS}ms and measurement {MEASUREMENT_MS}ms (scaled at and above {LARGE_SIZE_THRESHOLD})",
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
        bench_size(&mut suite, config, mode, len);
    }
    print!("{}", suite.report());

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
