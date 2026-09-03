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

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::{Complex32, Complex64};
use hermes_simd::{ProcessorBinding, ProcessorIndex};

use crate::application::execution::kernel::benchmark_kernels::rader_prime_forward;
use crate::application::execution::kernel::measurement_cores;

/// Primes routed through Rader. 101 is the item's recorded anomaly; the rest
/// are the lengths `rader::gather_sum_slice` names as its `f32` worst cases.
const PRIMES: &[usize] = &[67, 101, 113, 257];

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
            let mut reduced_out = reduced_source.clone();
            rader_prime_forward::<f32>(&mut reduced_out);
            assert_same_transform(n, &precise_out, &reduced_out);

            let mut precise_work = precise_source.clone();
            suite.run(BenchmarkCase::new(core, "rader/f64", n), || {
                precise_work.copy_from_slice(&precise_source);
                rader_prime_forward::<f64>(std::hint::black_box(&mut precise_work));
            });

            let mut reduced_work = reduced_source.clone();
            suite.run(BenchmarkCase::new(core, "rader/f32", n), || {
                reduced_work.copy_from_slice(&reduced_source);
                rader_prime_forward::<f32>(std::hint::black_box(&mut reduced_work));
            });
        }
        println!("RADER WIDTH cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
