//! Where does the `f32` prime-length anomaly live: in the kernel or the
//! dispatch around it?
//!
//! `ATLAS-APOLLO-F32-NONPOT-WIDTH` records `f32` at n = 101 slower than its
//! own `f64` on the full transform path (672 vs 585 ns on this tree,
//! `small_sizes_against_the_references_by_core_type`), while `rader_width`
//! shows the isolated Rader entry in the opposite order (804 vs 872). Both
//! instruments measure real quantities; their disagreement is the datum. The
//! full path for a prime is `exec_rader_forward` -> `rader_fft`, and the
//! entry probe calls `rader_prime_forward`, which reaches the same
//! `rader_runtime_impl` — but through a differently shaped call site, and
//! `try_static_rader` documents a measured `f32`-only codegen explosion at
//! that boundary (4275 instructions, 469 stack moves, against `f64`'s 1215
//! and zero).
//!
//! This probe reads the gap directly. Per prime and scalar it times, inside
//! one pinned run with identical batching:
//!
//! - **full** — the production plan and its forward entry;
//! - **entry** — the benchmark Rader entry point on the same buffer.
//!
//! If the anomaly is in the kernel, both rows inherit it and the per-scalar
//! `full - entry` gap is near zero for both scalars. If the gap is large for
//! `f32` and near zero (or negative) for `f64`, the dispatch boundary — not
//! the transform — is what `f32` pays for.
//!
//! Run optimized; see `rader_width` for why a debug build answers nothing
//! and why `median_ps` is already per-iteration:
//! `cargo test --release -p apollo-fft --lib prime_dispatch_by_core_type
//! -- --ignored --nocapture`.

use super::ProbeScalar;
use crate::application::execution::kernel::benchmark_kernels::rader_prime_forward;
use crate::application::execution::kernel::measurement_cores;
use apollo_bench::{BenchmarkCase, BenchmarkSuite};
use eunomia::{Complex32, Complex64};
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use std::time::Duration;

/// Primes that pin the two routes the dispatch can take below the Bluestein
/// power threshold: 101 and 251 are half-cyclic (smooth `p - 1`), 149 is
/// non-smooth at `p - 1 = 148` and takes Bluestein in both instruments. The
/// recorded anomaly is at 101; the other two say whether any boundary effect
/// is specific to it or general to primes.
const PRIMES: &[usize] = &[101, 149, 251];

/// The sweep's per-case budget: enough samples for a stable median at these
/// lengths, short enough that twelve cases across two cores stay well inside
/// the runner bound.
fn probe_config() -> apollo_bench::BenchmarkConfig {
    apollo_bench::BenchmarkConfig::try_with_budgets(
        Duration::from_millis(20),
        Duration::from_millis(80),
    )
    .expect("invariant: both budgets above are non-zero")
}

fn source<T: ProbeScalar>(n: usize) -> Vec<eunomia::Complex<T>> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            eunomia::Complex::new(
                T::from_precise((0.017 * x).sin()),
                T::from_precise(0.25 * (0.031 * x).cos()),
            )
        })
        .collect()
}

/// Pins the `f64` entry to the actual DFT, against `8 n eps`.
fn assert_entry_is_the_transform(n: usize, produced: &[Complex64], reference: &[Complex64]) {
    let tolerance = 8.0 * n as f64 * (f64::EPSILON / 2.0);
    let scale = reference
        .iter()
        .map(|value| value.re.abs().max(value.im.abs()))
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let gap = produced
        .iter()
        .zip(reference)
        .map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs()))
        .fold(0.0_f64, f64::max)
        / scale;
    assert!(
        gap < tolerance,
        "n={n}: the f64 entry departs from a direct DFT by {gap:e} relative"
    );
}

/// Confirms two arms compute the same transform, whichever normalization
/// each entry applies, by comparing them to each other rather than to a
/// separately normalized reference.
fn assert_same_transform(n: usize, label_a: &str, a: &[Complex64], label_b: &str, b: &[Complex64]) {
    let scale = a
        .iter()
        .map(|value| value.re.abs().max(value.im.abs()))
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let gap = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x.re - y.re).abs().max(x.im - y.im).abs())
        .fold(0.0_f64, f64::max)
        / scale;
    assert!(
        gap < 1e-9,
        "n={n}: {label_a} and {label_b} disagree by {gap:e} relative, so \
         no timing ratio between them is meaningful"
    );
}

fn assert_same_transform_f32(
    n: usize,
    label_a: &str,
    a: &[Complex32],
    label_b: &str,
    b: &[Complex64],
) {
    let eps = f64::from(f32::EPSILON) / 2.0;
    let tolerance = 4.0 * n as f64 * eps;
    let scale = b
        .iter()
        .map(|value| value.re.abs().max(value.im.abs()))
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let gap = b
        .iter()
        .zip(a)
        .map(|(x, y)| {
            let real = (x.re - f64::from(y.re)).abs();
            let imaginary = (x.im - f64::from(y.im)).abs();
            real.max(imaginary)
        })
        .fold(0.0_f64, f64::max)
        / scale;
    assert!(
        gap < tolerance,
        "n={n}: {label_a} departs from {label_b} by {gap:e} relative"
    );
}

fn direct_dft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .fold(Complex64::new(0.0, 0.0), |accumulator, (j, value)| {
                    let angle = -2.0 * std::f64::consts::PI * (k * j % n) as f64 / n as f64;
                    let (sin, cos) = angle.sin_cos();
                    Complex64::new(
                        accumulator.re + value.re * cos - value.im * sin,
                        accumulator.im + value.re * sin + value.im * cos,
                    )
                })
        })
        .collect()
}

#[test]
#[ignore = "measurement instrument for the f32 prime-dispatch gap"]
fn prime_dispatch_gap_by_core_type() {
    if cfg!(debug_assertions) {
        eprintln!("prime_dispatch: built without optimization; no timings reported.");
        return;
    }
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());

    for core in selection.cores() {
        let cpu = core.processor().get();
        let _binding = ProcessorBinding::bind(core.processor())
            .expect("measurement processor must be available");
        std::thread::yield_now();
        let landed = ProcessorIndex::current()
            .expect("Windows supports processor queries")
            .get();
        assert_eq!(landed, cpu, "processor binding must remain exact");
        let core = core.label();

        let mut suite = BenchmarkSuite::new(probe_config());
        for &n in PRIMES {
            let src64 = source::<f64>(n);
            let src32 = source::<f32>(n);

            let plan64 = crate::FftPlan1D::<f64>::new(
                crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
            );
            let plan32 = crate::FftPlan1D::<f32>::new(
                crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
            );

            // Correctness first; it also warms the plan, twiddle and order
            // caches, so no separate warm-up pass is needed.
            let mut full64 = src64.clone();
            plan64.forward_complex_slice_inplace(&mut full64);
            let mut entry64 = src64.clone();
            rader_prime_forward::<f64>(&mut entry64);
            assert_entry_is_the_transform(n, &entry64, &direct_dft(&src64));
            assert_same_transform(n, "full/f64", &full64, "entry/f64", &entry64);

            let mut full32 = src32.clone();
            plan32.forward_complex_slice_inplace(&mut full32);
            let mut entry32 = src32.clone();
            rader_prime_forward::<f32>(&mut entry32);
            assert_same_transform_f32(n, "full/f32", &full32, "full/f64", &full64);
            assert_same_transform_f32(n, "entry/f32", &entry32, "entry/f64", &entry64);

            // The reset is setup, not operation; each iteration gets a fresh
            // buffer built before the timer starts.
            suite.run_batched(
                BenchmarkCase::new(core, "full/f64", n),
                || src64.clone(),
                |work| {
                    plan64.forward_complex_slice_inplace(std::hint::black_box(work));
                    std::hint::black_box(work[0]);
                },
            );
            suite.run_batched(
                BenchmarkCase::new(core, "entry/f64", n),
                || src64.clone(),
                |work| {
                    rader_prime_forward::<f64>(std::hint::black_box(work));
                    std::hint::black_box(work[0]);
                },
            );
            suite.run_batched(
                BenchmarkCase::new(core, "full/f32", n),
                || src32.clone(),
                |work| {
                    plan32.forward_complex_slice_inplace(std::hint::black_box(work));
                    std::hint::black_box(work[0]);
                },
            );
            suite.run_batched(
                BenchmarkCase::new(core, "entry/f32", n),
                || src32.clone(),
                |work| {
                    rader_prime_forward::<f32>(std::hint::black_box(work));
                    std::hint::black_box(work[0]);
                },
            );
        }
        println!("PRIME DISPATCH cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
