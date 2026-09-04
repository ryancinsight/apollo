//! Does the `f32` Rader path run at its width?
//!
//! `ATLAS-APOLLO-F32-NONPOT-WIDTH` records `f32` at n = 101 measuring 674 ns
//! against its own `f64` at 579 — a prime-length transform where the narrower
//! scalar is *slower*, which no lane-width argument explains. That figure
//! predates the codelet-boundary work, so this re-measures it on the current
//! tree before anything is built on it.
//!
//! The two scalars are timed inside one pinned run per core, because the
//! quantity in question is the ratio between them: taking each from a separate
//! run would fold run-to-run variance into exactly the number being read.
//! Neither timing is believed until the two arms are shown to compute the same
//! transform.

//! # Run it optimized, and read `median_ps` as it stands
//!
//! Two things must be right before any number here means anything, and an
//! earlier revision of this file got both wrong in the same session:
//!
//! 1. **The profile.** Apollo defines `release`, `bench` and `bench-quick` but
//!    no `[profile.test]`, so plain `cargo nextest run` builds this at
//!    opt-level 0 and the timings describe unoptimized code — useless for a
//!    question about vector width. Run it as
//!    `cargo nextest run --cargo-profile bench-quick -E
//!    'test(rader_width_by_core_type)' --run-ignored all --no-capture`.
//!    The guard below refuses to report from a debug build rather than trusting
//!    the reader to remember.
//! 2. **The arithmetic.** `measurement::normalized_picoseconds` already divides
//!    each sample by `iterations_per_sample`, so the report's `median_ps` *is*
//!    per-iteration. Dividing it by the `iterations_per_sample` column again
//!    yields nonsense — it produced an apparent 0.4 ns for a 17-point
//!    transform, and because the two scalar arms calibrate to different
//!    iteration counts it also injected a spurious factor into the very ratio
//!    this probe exists to read.
//!
//! Those two mistakes together manufactured a phantom: an "unsound instrument"
//! reporting impossible times, and a run that appeared to confirm the recorded
//! n = 101 anomaly at 1.19. Neither was real.
//!
//! # What it measures, corrected
//!
//! Built optimized and read correctly, `f32` is never meaningfully slower than
//! `f64` at the Rader entry point on either core class. n = 101 measures 872 ns
//! for `f64` against 804 ns for `f32` (ratio 0.92) on a performance core, and
//! the larger lengths run 0.76-0.87. The recorded 674-against-579 ns anomaly
//! does not reproduce here.
//!
//! That is a statement about `rader_prime_forward`, not about the full
//! transform path the item's figures came from; the two are different
//! quantities, so this narrows the question rather than closing it.

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::{Complex32, Complex64};
use hermes_simd::{ProcessorBinding, ProcessorIndex};

use crate::application::execution::kernel::benchmark_kernels::rader_prime_forward;
use crate::application::execution::kernel::measurement_cores;

/// Primes routed through Rader, chosen so the convolution length `p - 1`
/// spans the radix mixes.
///
/// Rader convolves over `p - 1`, so that factorization — not `p` — is what the
/// inner transform sees. The first run found the `f32` penalty tracking how
/// close `p - 1` sits to a pure power of two, which this set is built to test
/// rather than assume: two pure powers (17, 257), two radix-3 (97, 193), two
/// radix-5 (41, 101), one radix-7 (113), and one carrying both odd radices
/// (151).
const PRIMES: &[usize] = &[17, 41, 97, 101, 113, 151, 193, 257];

/// The same deterministic signal both arms transform.
fn source(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

/// Confirms both scalars compute the same transform, within the error the
/// narrower one is entitled to.
///
/// The bound is the direct accumulation term, which dominates: a length-n
/// complex reduction accumulates at most `n` roundings per component, and a
/// complex multiply-add commits four of them, so `4 n eps` bounds the relative
/// departure. `f64`'s own error against the exact transform is smaller by the
/// ratio of the two epsilons (about 1e-9 here), so it serves as the oracle for
/// `f32` without needing a separate reference.
fn assert_same_transform(n: usize, precise: &[Complex64], reduced: &[Complex32]) {
    let eps = f64::from(f32::EPSILON) / 2.0;
    let tolerance = 4.0 * n as f64 * eps;
    let scale = precise
        .iter()
        .map(|value| value.re.abs().max(value.im.abs()))
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let gap = precise
        .iter()
        .zip(reduced)
        .map(|(a, b)| {
            let real = (a.re - f64::from(b.re)).abs();
            let imaginary = (a.im - f64::from(b.im)).abs();
            real.max(imaginary)
        })
        .fold(0.0_f64, f64::max)
        / scale;
    assert!(
        gap < tolerance,
        "n={n}: the two scalar arms disagree by {gap:e} relative against a \
         derived bound of {tolerance:e}, so they are not computing the same \
         transform and no ratio between their timings is meaningful"
    );
}

/// The transform this probe claims to be timing, computed directly.
///
/// `assert_same_transform` compares the two scalar arms against each other, so
/// it is satisfied by *any* shared behaviour — including a length that routes
/// somewhere else, or does nothing at all. This is the oracle that makes that
/// check live: it pins the `f64` arm to the actual DFT, and only then is `f64`
/// entitled to serve as the reference for `f32`.
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

/// Confirms the `f64` arm computed the transform, against `8 n eps` — the
/// direct summation's `n` roundings, doubled for the reference's own error and
/// again for the complex multiply-add.
fn assert_is_the_transform(n: usize, produced: &[Complex64], reference: &[Complex64]) {
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
        "n={n}: the f64 arm departs from a direct DFT by {gap:e} relative          against a derived bound of {tolerance:e}, so this length is not          running the transform this probe reports timings for"
    );
}

#[test]
#[ignore = "measurement instrument for the f32 Rader width question"]
fn rader_width_by_core_type() {
    // A debug build measures unoptimized code, which cannot answer a question
    // about vector width. Refuse rather than report a misleading number.
    if cfg!(debug_assertions) {
        eprintln!(
            "rader_width: built without optimization; re-run with --cargo-profile \n             bench-quick. No timings reported."
        );
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

        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        for &n in PRIMES {
            let precise_source = source(n);
            let reduced_source: Vec<Complex32> = precise_source
                .iter()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect();

            // Equivalence first; it also warms the twiddle and plan caches, so
            // no separate warm-up pass is needed to stay inside the runner
            // bound.
            let mut precise_out = precise_source.clone();
            rader_prime_forward::<f64>(&mut precise_out);
            assert_is_the_transform(n, &precise_out, &direct_dft(&precise_source));
            let mut reduced_out = reduced_source.clone();
            rader_prime_forward::<f32>(&mut reduced_out);
            assert_same_transform(n, &precise_out, &reduced_out);

            // The reset is setup, not operation. `run` would time the
            // `copy_from_slice` too, and `Complex64` copies twice the bytes of
            // `Complex32` -- charging the f64 arm for a wider memcpy and
            // biasing the very ratio these two cases exist to compare. Each
            // iteration gets its own buffer, built before the timer starts.
            suite.run_batched(
                BenchmarkCase::new(core, "rader/f64", n),
                || precise_source.clone(),
                |work| {
                    rader_prime_forward::<f64>(std::hint::black_box(work));
                    // Keeps the result live: an unread output is dead work the
                    // optimizer may drop entirely.
                    std::hint::black_box(work[0]);
                },
            );

            suite.run_batched(
                BenchmarkCase::new(core, "rader/f32", n),
                || reduced_source.clone(),
                |work| {
                    rader_prime_forward::<f32>(std::hint::black_box(work));
                    std::hint::black_box(work[0]);
                },
            );
        }
        println!("RADER WIDTH cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
