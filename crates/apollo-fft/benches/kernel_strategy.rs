//! Native Apollo benchmarks for FFT kernel strategies.

#![allow(missing_docs)]

#[cfg(feature = "kernel-strategy-bench")]
use apollo_bench::BenchmarkConfig;
use apollo_bench::{BenchmarkCase, BenchmarkSuite};
#[cfg(feature = "kernel-strategy-bench")]
use apollo_fft::application::execution::kernel::benchmark_kernels;
use apollo_fft::application::execution::kernel::{direct, fft_forward};
#[cfg(feature = "kernel-strategy-bench")]
use eunomia::Complex32;
use eunomia::{Complex, Complex64};
use half::f16;
use std::hint::black_box;
#[cfg(feature = "kernel-strategy-bench")]
use std::time::Duration;

fn signal(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

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
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect()
}

fn bench_fft_kernels(suite: &mut BenchmarkSuite) {
    for len in [16_usize, 32, 64, 128, 256] {
        let input = signal(len);
        if len <= 128 {
            suite.run(
                BenchmarkCase::new("fft_kernel_strategy", "direct_dft", len),
                || {
                    let output = direct::dft_forward(black_box(&input));
                    black_box(output);
                },
            );
        }

        suite.run(
            BenchmarkCase::new("fft_kernel_strategy", "generic_selector", len),
            || {
                let mut data = input.clone();
                fft_forward(black_box(&mut data));
                black_box(data);
            },
        );
    }

    for len in [31_usize, 63, 127] {
        let input = signal(len);
        suite.run(
            BenchmarkCase::new("fft_kernel_strategy", "generic_prime_inplace", len),
            || {
                let mut data = input.clone();
                fft_forward(black_box(&mut data));
                black_box(data);
            },
        );
    }

    for len in [64_usize, 96] {
        let input = signal_f16(len);
        suite.run(
            BenchmarkCase::new("fft_kernel_strategy", "mixed_precision_f16_auto", len),
            || {
                let mut data = input.clone();
                fft_forward(black_box(&mut data));
                black_box(data);
            },
        );
    }
}

#[cfg(feature = "kernel-strategy-bench")]
fn bench_prime_strategy(suite: &mut BenchmarkSuite, config: BenchmarkConfig) {
    for len in [19_usize, 29, 31, 37, 41, 43, 47, 53] {
        let input64 = signal(len);
        let mut rader64 = input64.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new(
                "fft_prime_strategy_rader_vs_winograd_pair",
                "rader_f64",
                len,
            ),
            || {
                rader64.copy_from_slice(&input64);
                benchmark_kernels::rader_prime_forward::<f64>(black_box(&mut rader64));
                black_box(&rader64);
            },
        );
        let mut winograd64 = input64.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new(
                "fft_prime_strategy_rader_vs_winograd_pair",
                "winograd_pair_f64",
                len,
            ),
            || {
                winograd64.copy_from_slice(&input64);
                benchmark_kernels::winograd_pair_prime_forward::<f64>(black_box(&mut winograd64));
                black_box(&winograd64);
            },
        );

        let input32 = signal32(len);
        let mut rader32 = input32.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new(
                "fft_prime_strategy_rader_vs_winograd_pair",
                "rader_f32",
                len,
            ),
            || {
                rader32.copy_from_slice(&input32);
                benchmark_kernels::rader_prime_forward::<f32>(black_box(&mut rader32));
                black_box(&rader32);
            },
        );
        let mut winograd32 = input32.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new(
                "fft_prime_strategy_rader_vs_winograd_pair",
                "winograd_pair_f32",
                len,
            ),
            || {
                winograd32.copy_from_slice(&input32);
                benchmark_kernels::winograd_pair_prime_forward::<f32>(black_box(&mut winograd32));
                black_box(&winograd32);
            },
        );
    }
}

fn main() {
    let mut suite = BenchmarkSuite::default();
    bench_fft_kernels(&mut suite);
    #[cfg(feature = "kernel-strategy-bench")]
    {
        let config = BenchmarkConfig::try_with_budgets(
            Duration::from_millis(100),
            Duration::from_millis(400),
        )
        .expect("invariant: literal benchmark budgets are non-zero");
        bench_prime_strategy(&mut suite, config);
    }
    suite.emit();
}
