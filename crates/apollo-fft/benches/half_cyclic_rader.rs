//! Native Apollo benchmarks for full-cyclic and half-cyclic Rader convolution.
//!
//! Run with `cargo bench -p apollo-fft --features kernel-strategy-bench --bench half_cyclic_rader`.

#![allow(missing_docs)]

#[cfg(feature = "kernel-strategy-bench")]
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
#[cfg(feature = "kernel-strategy-bench")]
use apollo_fft::application::execution::kernel::benchmark_kernels;
#[cfg(feature = "kernel-strategy-bench")]
use eunomia::{Complex32, Complex64};
#[cfg(feature = "kernel-strategy-bench")]
use std::hint::black_box;

#[cfg(feature = "kernel-strategy-bench")]
fn signal64(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let value = index as f64;
            Complex64::new((0.017 * value).sin(), 0.25 * (0.031 * value).cos())
        })
        .collect()
}

#[cfg(feature = "kernel-strategy-bench")]
fn signal32(len: usize) -> Vec<Complex32> {
    (0..len)
        .map(|index| {
            let value = index as f32;
            Complex32::new(
                (0.017_f32 * value).sin(),
                0.25_f32 * (0.031_f32 * value).cos(),
            )
        })
        .collect()
}

#[cfg(feature = "kernel-strategy-bench")]
fn bench_half_cyclic_rader(suite: &mut BenchmarkSuite, config: BenchmarkConfig) {
    // Geometric regimes cover the small, crossover, and large-prime paths.
    for len in [67_usize, 257, 521, 1031] {
        let input64 = signal64(len);
        let mut full64 = input64.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("rader_half_cyclic_vs_full_cyclic", "full_cyclic_f64", len),
            || {
                full64.copy_from_slice(&input64);
                benchmark_kernels::rader_full_cyclic_prime_forward::<f64>(black_box(&mut full64));
                black_box(&full64);
            },
        );
        let mut half64 = input64.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("rader_half_cyclic_vs_full_cyclic", "half_cyclic_f64", len),
            || {
                half64.copy_from_slice(&input64);
                benchmark_kernels::rader_half_cyclic_prime_forward::<f64>(black_box(&mut half64));
                black_box(&half64);
            },
        );
        let mut bluestein64 = input64.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("rader_half_cyclic_vs_full_cyclic", "bluestein_f64", len),
            || {
                bluestein64.copy_from_slice(&input64);
                benchmark_kernels::rader_bluestein_prime_forward::<f64>(black_box(
                    &mut bluestein64,
                ));
                black_box(&bluestein64);
            },
        );
        let mut automatic64 = input64.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("rader_half_cyclic_vs_full_cyclic", "auto_f64", len),
            || {
                automatic64.copy_from_slice(&input64);
                benchmark_kernels::rader_prime_forward::<f64>(black_box(&mut automatic64));
                black_box(&automatic64);
            },
        );

        let input32 = signal32(len);
        let mut full32 = input32.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("rader_half_cyclic_vs_full_cyclic", "full_cyclic_f32", len),
            || {
                full32.copy_from_slice(&input32);
                benchmark_kernels::rader_full_cyclic_prime_forward::<f32>(black_box(&mut full32));
                black_box(&full32);
            },
        );
        let mut half32 = input32.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("rader_half_cyclic_vs_full_cyclic", "half_cyclic_f32", len),
            || {
                half32.copy_from_slice(&input32);
                benchmark_kernels::rader_half_cyclic_prime_forward::<f32>(black_box(&mut half32));
                black_box(&half32);
            },
        );
        let mut bluestein32 = input32.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("rader_half_cyclic_vs_full_cyclic", "bluestein_f32", len),
            || {
                bluestein32.copy_from_slice(&input32);
                benchmark_kernels::rader_bluestein_prime_forward::<f32>(black_box(
                    &mut bluestein32,
                ));
                black_box(&bluestein32);
            },
        );
        let mut automatic32 = input32.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("rader_half_cyclic_vs_full_cyclic", "auto_f32", len),
            || {
                automatic32.copy_from_slice(&input32);
                benchmark_kernels::rader_prime_forward::<f32>(black_box(&mut automatic32));
                black_box(&automatic32);
            },
        );
    }
}

#[cfg(feature = "kernel-strategy-bench")]
fn bench_composite_radix_order(suite: &mut BenchmarkSuite, config: BenchmarkConfig) {
    let candidates: &[(&str, &[usize])] = &[
        ("r4_2_5_5", &[4, 2, 5, 5]),
        ("r4_5_5_2", &[4, 5, 5, 2]),
        ("r5_5_4_2", &[5, 5, 4, 2]),
        ("r5_4_5_2", &[5, 4, 5, 2]),
        ("r2_4_5_5", &[2, 4, 5, 5]),
    ];

    let input64 = signal64(200);
    for &(name, radices) in candidates {
        let mut buffer = input64.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("composite_radix_order", format!("{name}_f64"), 200),
            || {
                buffer.copy_from_slice(&input64);
                benchmark_kernels::composite_forward_with_radices::<f64>(
                    black_box(&mut buffer),
                    radices,
                );
                black_box(&buffer);
            },
        );
    }

    let input32 = signal32(200);
    for &(name, radices) in candidates {
        let mut buffer = input32.clone();
        suite.run_with_config(
            config,
            BenchmarkCase::new("composite_radix_order", format!("{name}_f32"), 200),
            || {
                buffer.copy_from_slice(&input32);
                benchmark_kernels::composite_forward_with_radices::<f32>(
                    black_box(&mut buffer),
                    radices,
                );
                black_box(&buffer);
            },
        );
    }
}

#[cfg(feature = "kernel-strategy-bench")]
fn main() -> Result<(), apollo_bench::BenchmarkModeError> {
    let mode = BenchmarkMode::from_environment()?;
    let config = mode.apply(BenchmarkConfig::regression());
    let mut suite = BenchmarkSuite::new(config);
    bench_half_cyclic_rader(&mut suite, config);
    bench_composite_radix_order(&mut suite, config);
    suite.emit();
    Ok(())
}

#[cfg(not(feature = "kernel-strategy-bench"))]
fn main() {
    eprintln!("enable the `kernel-strategy-bench` feature to run this benchmark");
}
