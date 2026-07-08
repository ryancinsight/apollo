//! Criterion benchmarks comparing Apollo FFT directly against RustFFT.
//!
//! This harness is intentionally owned by `apollo-fft` rather than
//! `apollo-validation` so the core CPU crate has a local performance contract.
//! It benchmarks clone-inclusive throughput, reusable-buffer latency, and
//! allocation behavior for planned execution.

#![allow(missing_docs)]

use apollo_fft::application::execution::kernel::FftPrecision;
use apollo_fft::{PlanCacheProvider, RealFftData, Shape1D};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use eunomia::{Complex32, Complex64};
use leto::Array1;
use rustfft::FftPlanner;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn signal_f64(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

fn signal_f32(len: usize) -> Vec<Complex32> {
    (0..len)
        .map(|i| {
            let x = i as f32;
            Complex32::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

fn reset_alloc_counter() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn allocation_count() -> u64 {
    ALLOCATIONS.load(Ordering::Relaxed)
}

fn selected_sizes(default: &[usize]) -> Vec<usize> {
    match std::env::var("APOLLO_FFT_BENCH_N") {
        Ok(raw) => raw
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(|part| {
                part.parse::<usize>()
                    .expect("APOLLO_FFT_BENCH_N entries must be positive integer FFT lengths")
            })
            .filter(|len| default.contains(len))
            .collect(),
        Err(std::env::VarError::NotPresent) => default.to_vec(),
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("APOLLO_FFT_BENCH_N must be valid Unicode")
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchProfile {
    sample_size: usize,
    measurement_time: Duration,
    warm_up_time: Duration,
}

impl BenchProfile {
    const fn quick() -> Self {
        Self {
            sample_size: 10,
            measurement_time: Duration::from_millis(150),
            warm_up_time: Duration::from_millis(20),
        }
    }

    const fn full() -> Self {
        Self {
            sample_size: 30,
            measurement_time: Duration::from_secs(2),
            warm_up_time: Duration::from_millis(250),
        }
    }
}

fn selected_bench_profile() -> BenchProfile {
    match std::env::var("APOLLO_FFT_BENCH_PROFILE") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "" | "quick" => BenchProfile::quick(),
            "full" => BenchProfile::full(),
            other => panic!("APOLLO_FFT_BENCH_PROFILE must be `quick` or `full`, got `{other}`"),
        },
        Err(std::env::VarError::NotPresent) => BenchProfile::quick(),
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("APOLLO_FFT_BENCH_PROFILE must be valid Unicode")
        }
    }
}

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    let profile = selected_bench_profile();
    group.sample_size(profile.sample_size);
    group.measurement_time(profile.measurement_time);
    group.warm_up_time(profile.warm_up_time);
}

fn bench_f64(c: &mut Criterion) {
    const SIZES: &[usize] = &[
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 25, 29, 31, 32, 33, 35, 36,
        37, 38, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 53, 56, 58, 59, 61, 63, 64, 67, 70, 71, 72,
        73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 88, 89, 90, 94, 96, 97, 98, 99, 100, 105, 108, 112,
        120, 125, 126, 128, 130, 135, 143, 144, 150, 154, 160, 165, 168, 175, 176, 180, 189, 192,
        196, 198, 200, 10_007, 32_768,
    ];
    let sizes = selected_sizes(SIZES);
    if sizes.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("apollo_fft_vs_rustfft_f64");
    configure_group(&mut group);

    for len in sizes {
        group.throughput(Throughput::Elements(len as u64));
        let input = signal_f64(len);

        group.bench_with_input(
            BenchmarkId::new("apollo_clone_inclusive", len),
            &input,
            |b, x| {
                b.iter(|| {
                    let mut data = x.clone();
                    Complex64::fft_forward(black_box(&mut data));
                    black_box(data);
                });
            },
        );

        let mut planner = FftPlanner::<f64>::new();
        let rustfft = planner.plan_fft_forward(len);
        let rustfft_input: Vec<rustfft::num_complex::Complex<f64>> = input
            .iter()
            .map(|z| rustfft::num_complex::Complex::new(z.re, z.im))
            .collect();
        let mut rustfft_scratch =
            vec![rustfft::num_complex::Complex::new(0.0, 0.0); rustfft.get_inplace_scratch_len()];

        group.bench_with_input(
            BenchmarkId::new("rustfft_clone_inclusive", len),
            &rustfft_input,
            |b, x| {
                b.iter(|| {
                    let mut data = x.clone();
                    rustfft.process_with_scratch(black_box(&mut data), &mut rustfft_scratch);
                    black_box(data);
                });
            },
        );

        if len.is_power_of_two() {
            let mut apollo_data = input.clone();
            Complex64::fft_forward(&mut apollo_data);
            apollo_data.copy_from_slice(&input);
            group.bench_function(BenchmarkId::new("apollo_reused_buffer_latency", len), |b| {
                b.iter(|| {
                    apollo_data.copy_from_slice(&input);
                    Complex64::fft_forward(black_box(&mut apollo_data));
                    black_box(&apollo_data);
                });
            });

            let mut rustfft_data = rustfft_input.clone();
            rustfft.process_with_scratch(&mut rustfft_data, &mut rustfft_scratch);
            rustfft_data.copy_from_slice(&rustfft_input);
            group.bench_function(
                BenchmarkId::new("rustfft_reused_buffer_latency", len),
                |b| {
                    b.iter(|| {
                        rustfft_data.copy_from_slice(&rustfft_input);
                        rustfft.process_with_scratch(
                            black_box(&mut rustfft_data),
                            &mut rustfft_scratch,
                        );
                        black_box(&rustfft_data);
                    });
                },
            );

            group.bench_function(BenchmarkId::new("apollo_zero_alloc_reused", len), |b| {
                b.iter_custom(|iters| {
                    apollo_data.copy_from_slice(&input);
                    Complex64::fft_forward(black_box(&mut apollo_data));
                    let elapsed = measure_zero_alloc(iters, || {
                        apollo_data.copy_from_slice(&input);
                        Complex64::fft_forward(black_box(&mut apollo_data));
                    });
                    assert_eq!(
                        allocation_count(),
                        0,
                        "Apollo planned f64 allocated in hot loop"
                    );
                    elapsed
                });
            });

            group.bench_function(BenchmarkId::new("rustfft_zero_alloc_reused", len), |b| {
                b.iter_custom(|iters| {
                    let elapsed = measure_zero_alloc(iters, || {
                        rustfft_data.copy_from_slice(&rustfft_input);
                        rustfft.process_with_scratch(
                            black_box(&mut rustfft_data),
                            &mut rustfft_scratch,
                        );
                    });
                    assert_eq!(
                        allocation_count(),
                        0,
                        "RustFFT planned f64 allocated in hot loop"
                    );
                    elapsed
                });
            });

            let rustfft_inverse = planner.plan_fft_inverse(len);
            let mut rustfft_inverse_scratch = vec![
                rustfft::num_complex::Complex::new(0.0, 0.0);
                rustfft_inverse.get_inplace_scratch_len()
            ];

            group.bench_function(
                BenchmarkId::new("apollo_inverse_unnorm_zero_alloc_reused", len),
                |b| {
                    b.iter_custom(|iters| {
                        apollo_data.copy_from_slice(&input);
                        Complex64::fft_inverse_unnorm(black_box(&mut apollo_data));
                        let elapsed = measure_zero_alloc(iters, || {
                            apollo_data.copy_from_slice(&input);
                            Complex64::fft_inverse_unnorm(black_box(&mut apollo_data));
                        });
                        assert_eq!(
                            allocation_count(),
                            0,
                            "Apollo generic inverse f64 allocated in hot loop"
                        );
                        elapsed
                    });
                },
            );

            group.bench_function(
                BenchmarkId::new("rustfft_inverse_zero_alloc_reused", len),
                |b| {
                    b.iter_custom(|iters| {
                        let elapsed = measure_zero_alloc(iters, || {
                            rustfft_data.copy_from_slice(&rustfft_input);
                            rustfft_inverse.process_with_scratch(
                                black_box(&mut rustfft_data),
                                &mut rustfft_inverse_scratch,
                            );
                        });
                        assert_eq!(
                            allocation_count(),
                            0,
                            "RustFFT planned inverse f64 allocated in hot loop"
                        );
                        elapsed
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_f32(c: &mut Criterion) {
    const SIZES: &[usize] = &[
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 25, 29, 31, 32, 33, 35, 36,
        37, 38, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 53, 56, 58, 59, 61, 63, 64, 67, 70, 71, 72,
        73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 88, 89, 90, 94, 96, 97, 98, 99, 100, 105, 108, 112,
        120, 125, 126, 128, 130, 135, 143, 144, 150, 154, 160, 165, 168, 175, 176, 180, 189, 192,
        196, 198, 200, 10_007, 32_768,
    ];
    let sizes = selected_sizes(SIZES);
    if sizes.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("apollo_fft_vs_rustfft_f32");
    configure_group(&mut group);

    for len in sizes {
        group.throughput(Throughput::Elements(len as u64));
        let input = signal_f32(len);

        group.bench_with_input(
            BenchmarkId::new("apollo_clone_inclusive", len),
            &input,
            |b, x| {
                b.iter(|| {
                    let mut data = x.clone();
                    Complex32::fft_forward(black_box(&mut data));
                    black_box(data);
                });
            },
        );

        let mut planner = FftPlanner::<f32>::new();
        let rustfft = planner.plan_fft_forward(len);
        let rustfft_input: Vec<rustfft::num_complex::Complex<f32>> = input
            .iter()
            .map(|z| rustfft::num_complex::Complex::new(z.re, z.im))
            .collect();
        let mut rustfft_scratch =
            vec![rustfft::num_complex::Complex::new(0.0, 0.0); rustfft.get_inplace_scratch_len()];

        group.bench_with_input(
            BenchmarkId::new("rustfft_clone_inclusive", len),
            &rustfft_input,
            |b, x| {
                b.iter(|| {
                    let mut data = x.clone();
                    rustfft.process_with_scratch(black_box(&mut data), &mut rustfft_scratch);
                    black_box(data);
                });
            },
        );

        if len.is_power_of_two() {
            let mut apollo_data = input.clone();
            Complex32::fft_forward(&mut apollo_data);
            apollo_data.copy_from_slice(&input);
            group.bench_function(BenchmarkId::new("apollo_reused_buffer_latency", len), |b| {
                b.iter(|| {
                    apollo_data.copy_from_slice(&input);
                    Complex32::fft_forward(black_box(&mut apollo_data));
                    black_box(&apollo_data);
                });
            });

            let mut rustfft_data = rustfft_input.clone();
            rustfft.process_with_scratch(&mut rustfft_data, &mut rustfft_scratch);
            rustfft_data.copy_from_slice(&rustfft_input);
            group.bench_function(
                BenchmarkId::new("rustfft_reused_buffer_latency", len),
                |b| {
                    b.iter(|| {
                        rustfft_data.copy_from_slice(&rustfft_input);
                        rustfft.process_with_scratch(
                            black_box(&mut rustfft_data),
                            &mut rustfft_scratch,
                        );
                        black_box(&rustfft_data);
                    });
                },
            );

            group.bench_function(BenchmarkId::new("apollo_zero_alloc_reused", len), |b| {
                b.iter_custom(|iters| {
                    apollo_data.copy_from_slice(&input);
                    Complex32::fft_forward(black_box(&mut apollo_data));
                    let elapsed = measure_zero_alloc(iters, || {
                        apollo_data.copy_from_slice(&input);
                        Complex32::fft_forward(black_box(&mut apollo_data));
                    });
                    assert_eq!(
                        allocation_count(),
                        0,
                        "Apollo planned f32 allocated in hot loop"
                    );
                    elapsed
                });
            });

            group.bench_function(BenchmarkId::new("rustfft_zero_alloc_reused", len), |b| {
                b.iter_custom(|iters| {
                    let elapsed = measure_zero_alloc(iters, || {
                        rustfft_data.copy_from_slice(&rustfft_input);
                        rustfft.process_with_scratch(
                            black_box(&mut rustfft_data),
                            &mut rustfft_scratch,
                        );
                    });
                    assert_eq!(
                        allocation_count(),
                        0,
                        "RustFFT planned f32 allocated in hot loop"
                    );
                    elapsed
                });
            });

            let rustfft_inverse = planner.plan_fft_inverse(len);
            let mut rustfft_inverse_scratch = vec![
                rustfft::num_complex::Complex::new(0.0, 0.0);
                rustfft_inverse.get_inplace_scratch_len()
            ];

            group.bench_function(
                BenchmarkId::new("apollo_inverse_unnorm_zero_alloc_reused", len),
                |b| {
                    b.iter_custom(|iters| {
                        apollo_data.copy_from_slice(&input);
                        Complex32::fft_inverse_unnorm(black_box(&mut apollo_data));
                        let elapsed = measure_zero_alloc(iters, || {
                            apollo_data.copy_from_slice(&input);
                            Complex32::fft_inverse_unnorm(black_box(&mut apollo_data));
                        });
                        assert_eq!(
                            allocation_count(),
                            0,
                            "Apollo generic inverse f32 allocated in hot loop"
                        );
                        elapsed
                    });
                },
            );

            group.bench_function(
                BenchmarkId::new("rustfft_inverse_zero_alloc_reused", len),
                |b| {
                    b.iter_custom(|iters| {
                        let elapsed = measure_zero_alloc(iters, || {
                            rustfft_data.copy_from_slice(&rustfft_input);
                            rustfft_inverse.process_with_scratch(
                                black_box(&mut rustfft_data),
                                &mut rustfft_inverse_scratch,
                            );
                        });
                        assert_eq!(
                            allocation_count(),
                            0,
                            "RustFFT planned inverse f32 allocated in hot loop"
                        );
                        elapsed
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_six_step_f32(c: &mut Criterion) {
    const SIZES: &[usize] = &[5 * 1024, 7 * 1024, 11 * 512];
    let sizes = selected_sizes(SIZES);
    if sizes.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("apollo_six_step_f32_vs_rustfft");
    configure_group(&mut group);

    for len in sizes {
        group.throughput(Throughput::Elements(len as u64));
        let input =
            Array1::from_shape_vec([len], signal_f32(len).into_iter().map(|z| z.re).collect())
                .expect("bench input length matches shape");
        let plan = f32::get_1d_plan(Shape1D::new(len).expect("bench length must be non-zero"));

        group.bench_with_input(
            BenchmarkId::new("apollo_public_six_step", len),
            &input,
            |b, x| {
                b.iter(|| {
                    black_box(<f32 as RealFftData>::forward_1d(
                        plan.as_ref(),
                        black_box(x),
                    ));
                });
            },
        );

        let mut apollo_output = Array1::<Complex32>::zeros([len]);
        group.bench_with_input(
            BenchmarkId::new("apollo_caller_owned_six_step", len),
            &input,
            |b, x| {
                b.iter(|| {
                    <f32 as RealFftData>::forward_1d_into(
                        plan.as_ref(),
                        black_box(x),
                        &mut apollo_output,
                    );
                    black_box(&apollo_output);
                });
            },
        );

        group.bench_function(BenchmarkId::new("apollo_zero_alloc_six_step", len), |b| {
            b.iter_custom(|iters| {
                <f32 as RealFftData>::forward_1d_into(
                    plan.as_ref(),
                    black_box(&input),
                    &mut apollo_output,
                );
                let elapsed = measure_zero_alloc(iters, || {
                    <f32 as RealFftData>::forward_1d_into(
                        plan.as_ref(),
                        black_box(&input),
                        &mut apollo_output,
                    );
                });
                assert_eq!(
                    allocation_count(),
                    0,
                    "Apollo six-step f32 allocated in the caller-owned hot loop"
                );
                elapsed
            });
        });

        let rustfft_input: Vec<rustfft::num_complex::Complex<f32>> = input
            .iter()
            .map(|&x| rustfft::num_complex::Complex::new(x, 0.0))
            .collect();
        let mut planner = FftPlanner::<f32>::new();
        let rustfft = planner.plan_fft_forward(len);
        let mut rustfft_data = rustfft_input.clone();
        let mut rustfft_scratch =
            vec![rustfft::num_complex::Complex::new(0.0, 0.0); rustfft.get_inplace_scratch_len()];

        group.bench_function(BenchmarkId::new("rustfft_caller_owned", len), |b| {
            b.iter(|| {
                rustfft_data.copy_from_slice(&rustfft_input);
                rustfft.process_with_scratch(black_box(&mut rustfft_data), &mut rustfft_scratch);
                black_box(&rustfft_data);
            });
        });

        group.bench_function(BenchmarkId::new("rustfft_zero_alloc", len), |b| {
            b.iter_custom(|iters| {
                let elapsed = measure_zero_alloc(iters, || {
                    rustfft_data.copy_from_slice(&rustfft_input);
                    rustfft
                        .process_with_scratch(black_box(&mut rustfft_data), &mut rustfft_scratch);
                });
                assert_eq!(
                    allocation_count(),
                    0,
                    "RustFFT planned f32 allocated in the caller-owned hot loop"
                );
                elapsed
            });
        });
    }

    group.finish();
}

fn measure_zero_alloc(mut iters: u64, mut f: impl FnMut()) -> Duration {
    reset_alloc_counter();
    let start = Instant::now();
    while iters > 0 {
        f();
        iters -= 1;
    }
    start.elapsed()
}

criterion_group!(benches, bench_f64, bench_f32, bench_six_step_f32);
criterion_main!(benches);
