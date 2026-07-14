//! Sub-minute benchmarks for the recursive prime-CT dispatch path.
//!
//! ## Goal
//!
//! Measure the `fft_forward` dispatch path (which routes through the recursive
//! prime-CT engine for N ≤ PRIME_CT_MAX_N) and confirm sub-microsecond latency
//! for small composite sizes across powers-of-two and smooth composites.

#![allow(missing_docs)]

use apollo_fft::application::execution::kernel::FftPrecision;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use eunomia::Complex64;
use std::hint::black_box;

fn signal(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|k| Complex64::new((0.271 * k as f64).sin(), 0.349 * (0.113 * k as f64).cos()))
        .collect()
}

fn bench_sizes(c: &mut Criterion, label: &str, sizes: &[usize]) {
    let mut group = c.benchmark_group(label);
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_millis(800));

    for &n in sizes {
        let input = signal(n);

        // In-place on pre-allocated buffer — measures the kernel cost only.
        group.bench_with_input(
            BenchmarkId::new("radix_composite_inplace", n),
            &input,
            |b, inp| {
                let mut buf = inp.clone();
                b.iter(|| {
                    Complex64::fft_forward(black_box(&mut buf));
                    black_box(&buf);
                    buf.copy_from_slice(inp);
                });
            },
        );

        // Clone-inclusive — measures allocation + kernel.
        group.bench_with_input(
            BenchmarkId::new("radix_composite_clone_inclusive", n),
            &input,
            |b, inp| {
                b.iter(|| {
                    let mut buf = inp.clone();
                    Complex64::fft_forward(black_box(&mut buf));
                    black_box(buf);
                });
            },
        );
    }

    group.finish();
}

fn bench_pot(c: &mut Criterion) {
    bench_sizes(c, "radix_composite_powers_of_two", &[4, 8, 16, 32, 64]);
}

fn bench_smooth(c: &mut Criterion) {
    bench_sizes(
        c,
        "radix_composite_smooth_composites",
        &[
            6, 9, 10, 12, 14, 15, 18, 20, 21, 24, 25, 27, 28, 30, 36, 45, 49, 50, 60, 63,
        ],
    );
}

fn bench_two_by_prime(c: &mut Criterion) {
    bench_sizes(
        c,
        "two_by_prime_coprime_composites",
        &[38, 58, 62, 74, 82, 86, 94, 106],
    );
}

criterion_group!(benches, bench_pot, bench_smooth, bench_two_by_prime);
criterion_main!(benches);
