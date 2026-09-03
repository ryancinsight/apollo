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

//! # This instrument is not yet sound. Do not read verdicts from it.
//!
//! Measured on a quiet host (12% CPU) it still reports **0.4 ns per iteration
//! for a 17-point transform** — about one cycle — while `assert_is_the_transform`
//! confirms that same length computes a correct DFT. Correct output at an
//! impossible time means the timing, not the math, is wrong.
//!
//! Three hardening passes did not fix it: a direct-DFT oracle (which did make
//! the equivalence check live, and is worth keeping), consuming the output so
//! it cannot be dead, and blinding the input so the transform cannot be hoisted
//! as a loop-invariant pure function. Across those runs the n = 101 performance
//! -core ratio read 1.19, then 0.97, then 1.25 — the quantity the probe exists
//! to measure is not reproducible run to run.
//!
//! So no verdict here is publishable, including the ones that happen to agree
//! with the item: a single run of this probe appeared to confirm the recorded
//! n = 101 anomaly at 1.19 against a recorded 1.16, and to localize it to
//! performance cores. The next run contradicted it. An instrument that reports
//! one cycle for a 17-point transform cannot be trusted selectively on the
//! lengths whose answers look plausible.
//!
//! **Open question, and the thing to fix first:** where the sub-nanosecond
//! reading comes from. Either `BenchmarkConfig::regression()` mis-calibrates
//! `iterations_per_sample` for sub-microsecond cases — it reported 3352
//! iterations in a 1.13 ms sample — or something upstream is returning a cached
//! spectrum for a repeated input. The second would be a correctness question,
//! not just a measurement one, which is why this is recorded rather than
//! deleted.

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

            let mut precise_work = precise_source.clone();
            suite.run(BenchmarkCase::new(core, "rader/f64", n), || {
                precise_work.copy_from_slice(std::hint::black_box(&precise_source));
                rader_prime_forward::<f64>(std::hint::black_box(&mut precise_work));
                // Both ends of the loop must be opaque. Hiding only the
                // output pointer leaves the transform a pure function of an
                // input that never changes, so it hoists clean out of the timed
                // loop -- which is what produced sub-nanosecond readings for a
                // 17-point transform. Blinding the source defeats the
                // invariance; reading an element keeps the result live.
                std::hint::black_box(precise_work[0]);
            });

            let mut reduced_work = reduced_source.clone();
            suite.run(BenchmarkCase::new(core, "rader/f32", n), || {
                reduced_work.copy_from_slice(std::hint::black_box(&reduced_source));
                rader_prime_forward::<f32>(std::hint::black_box(&mut reduced_work));
                // Both ends of the loop must be opaque. Hiding only the
                // output pointer leaves the transform a pure function of an
                // input that never changes, so it hoists clean out of the timed
                // loop -- which is what produced sub-nanosecond readings for a
                // 17-point transform. Blinding the source defeats the
                // invariance; reading an element keeps the result live.
                std::hint::black_box(reduced_work[0]);
            });
        }
        println!("RADER WIDTH cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
