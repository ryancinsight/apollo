//! Native Apollo benchmarks for the recursive prime-CT dispatch path.
//!
//! Measures the `fft_forward` dispatch path for powers-of-two, smooth
//! composites, and two-by-prime composites without deleting either the
//! in-place or clone-inclusive workload.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use apollo_fft::application::execution::kernel::FftPrecision;
use eunomia::Complex64;
use std::hint::black_box;

fn signal(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|k| Complex64::new((0.271 * k as f64).sin(), 0.349 * (0.113 * k as f64).cos()))
        .collect()
}

fn bench_sizes(suite: &mut BenchmarkSuite, group: &str, sizes: &[usize]) {
    for &n in sizes {
        let input = signal(n);
        let mut in_place = input.clone();
        suite.run(
            BenchmarkCase::new(group, "radix_composite_inplace", n),
            || {
                Complex64::fft_forward(black_box(&mut in_place));
                black_box(&in_place);
                in_place.copy_from_slice(&input);
            },
        );

        suite.run(
            BenchmarkCase::new(group, "radix_composite_clone_inclusive", n),
            || {
                let mut clone = input.clone();
                Complex64::fft_forward(black_box(&mut clone));
                black_box(clone);
            },
        );
    }
}

fn main() -> Result<(), apollo_bench::BenchmarkModeError> {
    let mode = BenchmarkMode::from_environment()?;
    let config = mode.apply(BenchmarkConfig::regression());
    let mut suite = BenchmarkSuite::new(config);
    // Each family uses geometric representatives for its distinct dispatch regime.
    bench_sizes(&mut suite, "radix_composite_powers_of_two", &[4, 16, 64]);
    bench_sizes(
        &mut suite,
        "radix_composite_smooth_composites",
        &[6, 15, 36, 63],
    );
    bench_sizes(
        &mut suite,
        "two_by_prime_coprime_composites",
        &[38, 62, 106],
    );
    suite.emit();
    Ok(())
}
