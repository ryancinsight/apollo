//! The two radix tables against each other, order isolated from everything
//! else.
//!
//! `static_prime23_radices` and `cached_prime23_radices` factor a composite
//! length identically and order the factors differently — 154 of the 162
//! lengths both carry. The static table leads with the power-of-two radices
//! (`[4, 13]`, `[4, 2, 7]`), the cached one with the largest factor
//! (`[13, 4]`, `[7, 4, 2]`). Which is faster has measured both ways, so this
//! runs the same kernel over the same data with only the order differing.

use crate::application::execution::kernel::benchmark_kernels::composite_forward_with_radices;
use crate::application::execution::kernel::measurement_cores;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use hermes_simd::{ProcessorBinding, ProcessorIndex};

/// Lengths spanning the divergent set: every factor shape it contains, from
/// the smallest to past the cache-resident sizes.
const SURVEY: &[usize] = &[
    12, 24, 36, 48, 60, 72, 90, 96, 100, 120, 180, 240, 360, 384, 720, 1008,
];

#[cfg(test)]
fn both_orders(suite: &mut BenchmarkSuite, core: &str) {
    use crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices;
    use crate::application::execution::kernel::mixed_radix::dispatch::static_prime23_radices;
    use eunomia::Complex32;

    for &n in SURVEY {
        let (Some(statik), Some(cached)) = (static_prime23_radices(n), cached_prime23_radices(n))
        else {
            continue;
        };
        if statik == cached.as_ref() {
            continue;
        }
        let source: Vec<Complex32> = (0..n)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.017_f32 * x).sin(), 0.25 * (0.031_f32 * x).cos())
            })
            .collect();

        let mut work = source.clone();
        suite.run(BenchmarkCase::new(core, "pot-first", n), || {
            work.copy_from_slice(&source);
            composite_forward_with_radices(std::hint::black_box(&mut work), statik);
        });

        let mut work_cached = source.clone();
        suite.run(BenchmarkCase::new(core, "largest-first", n), || {
            work_cached.copy_from_slice(&source);
            composite_forward_with_radices(std::hint::black_box(&mut work_cached), &cached);
        });
    }
}

/// The plan's hand-listed entries against the derived order.
///
/// `FftPlan1D::new` carries a short list of lengths with explicit radix
/// orders, written before the derivation was corrected. Most of them already
/// agree with it; these three do not, so each is measured rather than assumed
/// either way.
#[cfg(test)]
fn hand_listed_against_derived(suite: &mut BenchmarkSuite, core: &str) {
    use eunomia::Complex32;

    for (n, hand, derived) in [
        (180usize, &[5usize, 3, 3, 4][..], &[4usize, 3, 3, 5][..]),
        (176, &[11, 4, 4][..], &[4, 4, 11][..]),
        (385, &[11, 5, 7][..], &[5, 7, 11][..]),
    ] {
        let source: Vec<Complex32> = (0..n)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.017_f32 * x).sin(), 0.25 * (0.031_f32 * x).cos())
            })
            .collect();

        let mut work = source.clone();
        suite.run(BenchmarkCase::new(core, "hand-listed", n), || {
            work.copy_from_slice(&source);
            composite_forward_with_radices(std::hint::black_box(&mut work), hand);
        });

        let mut work_derived = source.clone();
        suite.run(BenchmarkCase::new(core, "derived", n), || {
            work_derived.copy_from_slice(&source);
            composite_forward_with_radices(std::hint::black_box(&mut work_derived), derived);
        });
    }
}

#[test]
#[ignore = "measurement instrument for the radix-order disagreement"]
fn radix_order_by_core_type() {
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
        let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
        both_orders(&mut warmup, core);
        hand_listed_against_derived(&mut warmup, core);
        drop(warmup);
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        both_orders(&mut suite, core);
        hand_listed_against_derived(&mut suite, core);
        println!("RADIX cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
