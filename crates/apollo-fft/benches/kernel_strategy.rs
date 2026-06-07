//! Criterion benchmarks for Apollo FFT kernel strategies.

#![allow(missing_docs)]

#[cfg(feature = "kernel-strategy-bench")]
use apollo_fft::application::execution::kernel::benchmark_kernels;
use apollo_fft::application::execution::kernel::{direct, fft_forward};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;
#[cfg(feature = "kernel-strategy-bench")]
use num_complex::Complex32;
use num_complex::{Complex, Complex64};

/// Generate a deterministic complex sinusoidal test signal of the given length.
fn signal(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

/// Deterministic f16-complex signal used by mixed-precision selector benchmarks.
fn signal_f16(len: usize) -> Vec<Complex<f16>> {
    (0..len)
        .map(|index| {
            let x = index as f32;
            Complex::new(
                f16::from_f32((0.017 * x).sin()),
                f16::from_f32(0.25 * (0.031 * x).cos()),
            )
        })
        .collect()
}

#[cfg(feature = "kernel-strategy-bench")]
fn signal32(len: usize) -> Vec<Complex32> {
    signal(len)
        .into_iter()
        .map(|z| Complex32::new(z.re as f32, z.im as f32))
        .collect()
}

/// Benchmark direct-DFT, mixed-radix, and auto-selector kernels.
fn bench_fft_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_kernel_strategy");

    for len in [16usize, 32, 64, 128, 256] {
        let input = signal(len);
        if len <= 128 {
            group.bench_with_input(
                BenchmarkId::new("direct_dft", len),
                &input,
                |bench, input| {
                    bench.iter(|| {
                        let output = direct::dft_forward(black_box(input));
                        black_box(output);
                    });
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("generic_selector", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut data = input.clone();
                    fft_forward(black_box(&mut data));
                    black_box(data);
                });
            },
        );
    }

    for len in [31usize, 63, 127] {
        let input = signal(len);
        group.bench_with_input(
            BenchmarkId::new("generic_prime_inplace", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut data = input.clone();
                    fft_forward(black_box(&mut data));
                    black_box(data);
                });
            },
        );
    }

    for len in [64usize, 96] {
        let input = signal_f16(len);
        group.bench_with_input(
            BenchmarkId::new("mixed_precision_f16_auto", len),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let mut data = input.clone();
                    fft_forward(black_box(&mut data));
                    black_box(data);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "kernel-strategy-bench")]
fn bench_prime_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_prime_strategy_rader_vs_winograd_pair");
    group.warm_up_time(std::time::Duration::from_millis(100));
    group.measurement_time(std::time::Duration::from_millis(400));

    for len in [19usize, 29, 31, 37, 41, 43, 47, 53] {
        let input64 = signal(len);
        group.bench_with_input(
            BenchmarkId::new("rader_f64", len),
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
        group.bench_with_input(
            BenchmarkId::new("winograd_pair_f64", len),
            &input64,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::winograd_pair_prime_forward::<f64>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );

        let input32 = signal32(len);
        group.bench_with_input(
            BenchmarkId::new("rader_f32", len),
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
        group.bench_with_input(
            BenchmarkId::new("winograd_pair_f32", len),
            &input32,
            |bench, input| {
                let mut buf = input.clone();
                bench.iter(|| {
                    buf.copy_from_slice(input);
                    benchmark_kernels::winograd_pair_prime_forward::<f32>(black_box(&mut buf));
                    black_box(&buf);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "kernel-strategy-bench")]
criterion_group!(benches, bench_fft_kernels, bench_prime_strategy);
#[cfg(not(feature = "kernel-strategy-bench"))]
criterion_group!(benches, bench_fft_kernels);
criterion_main!(benches);
