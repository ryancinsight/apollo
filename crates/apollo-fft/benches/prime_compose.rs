//! Native Apollo benchmarks for the recursive prime-CT dispatch path.
//!
//! Measures the `fft_forward` dispatch path for powers-of-two, smooth
//! composites, and two-by-prime composites without deleting either the
//! in-place or clone-inclusive workload.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use apollo_fft::application::execution::kernel::FftPrecision;
use eunomia::Complex64;
use std::hint::black_box;
use std::time::Duration;

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

fn main() {
    let config =
        BenchmarkConfig::try_with_budgets(Duration::from_millis(200), Duration::from_millis(800))
            .expect("invariant: literal benchmark budgets are non-zero");
    let mut suite = BenchmarkSuite::new(config);
    bench_sizes(
        &mut suite,
        "radix_composite_powers_of_two",
        &[4, 8, 16, 32, 64],
    );
    bench_sizes(
        &mut suite,
        "radix_composite_smooth_composites",
        &[
            6, 9, 10, 12, 14, 15, 18, 20, 21, 24, 25, 27, 28, 30, 36, 45, 49, 50, 60, 63,
        ],
    );
    bench_sizes(
        &mut suite,
        "two_by_prime_coprime_composites",
        &[38, 58, 62, 74, 82, 86, 94, 106],
    );
    suite.emit();
}
