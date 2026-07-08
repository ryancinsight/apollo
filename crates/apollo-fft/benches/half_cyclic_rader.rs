//! Full-cyclic versus half-cyclic Rader convolution benchmarks.
//!
//! Run with:
//! `cargo bench -p apollo-fft --features kernel-strategy-bench --bench half_cyclic_rader`.

#![allow(missing_docs)]

#[cfg(feature = "kernel-strategy-bench")]
use apollo_fft::application::execution::kernel::benchmark_kernels;
#[cfg(feature = "kernel-strategy-bench")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "kernel-strategy-bench")]
use eunomia::{Complex32, Complex64};

#[cfg(feature = "kernel-strategy-bench")]
fn signal64(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

#[cfg(feature = "kernel-strategy-bench")]
fn signal32(len: usize) -> Vec<Complex32> {
    (0..len)
        .map(|index| {
            let x = index as f32;
            Complex32::new((0.017_f32 * x).sin(), 0.25_f32 * (0.031_f32 * x).cos())
        })
        .collect()
}

#[cfg(feature = "kernel-strategy-bench")]
fn bench_half_cyclic_rader(c: &mut Criterion) {
    let mut group = c.benchmark_group("rader_half_cyclic_vs_full_cyclic");
    group.warm_up_time(std::time::Duration::from_millis(150));
    group.measurement_time(std::time::Duration::from_millis(500));

    for len in [67usize, 101, 257, 271, 337, 521, 1031] {
        let input64 = signal64(len);
        group.bench_with_input(
            BenchmarkId::new("full_cyclic_f64", len),
            &input64,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::rader_full_cyclic_prime_forward::<f64>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("half_cyclic_f64", len),
            &input64,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::rader_half_cyclic_prime_forward::<f64>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("bluestein_f64", len),
            &input64,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::rader_bluestein_prime_forward::<f64>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("auto_f64", len),
            &input64,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::rader_prime_forward::<f64>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );

        let input32 = signal32(len);
        group.bench_with_input(
            BenchmarkId::new("full_cyclic_f32", len),
            &input32,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::rader_full_cyclic_prime_forward::<f32>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("half_cyclic_f32", len),
            &input32,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::rader_half_cyclic_prime_forward::<f32>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("bluestein_f32", len),
            &input32,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::rader_bluestein_prime_forward::<f32>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("auto_f32", len),
            &input32,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::rader_prime_forward::<f32>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "kernel-strategy-bench")]
fn bench_composite_radix_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("composite_radix_order");
    group.warm_up_time(std::time::Duration::from_millis(150));
    group.measurement_time(std::time::Duration::from_millis(500));

    let candidates: &[(&str, &[usize])] = &[
        ("r4_2_5_5", &[4, 2, 5, 5]),
        ("r4_5_5_2", &[4, 5, 5, 2]),
        ("r5_5_4_2", &[5, 5, 4, 2]),
        ("r5_4_5_2", &[5, 4, 5, 2]),
        ("r2_4_5_5", &[2, 4, 5, 5]),
    ];

    let input64 = signal64(200);
    for &(name, radices) in candidates {
        group.bench_with_input(
            BenchmarkId::new(format!("{name}_f64"), 200),
            &input64,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::composite_forward_with_radices::<f64>(
                        black_box(&mut buf),
                        radices,
                    );
                    black_box(&buf);
                });
            },
        );
    }

    let input32 = signal32(200);
    for &(name, radices) in candidates {
        group.bench_with_input(
            BenchmarkId::new(format!("{name}_f32"), 200),
            &input32,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::composite_forward_with_radices::<f32>(
                        black_box(&mut buf),
                        radices,
                    );
                    black_box(&buf);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "kernel-strategy-bench")]
criterion_group!(
    benches,
    bench_half_cyclic_rader,
    bench_composite_radix_order
);
#[cfg(feature = "kernel-strategy-bench")]
criterion_main!(benches);

#[cfg(not(feature = "kernel-strategy-bench"))]
fn main() {
    eprintln!("enable the `kernel-strategy-bench` feature to run this benchmark");
}
