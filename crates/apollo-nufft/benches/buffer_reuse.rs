//! Native Apollo benchmarks for provider-backed NUFFT buffer reuse.
//!
//! Each case retains the existing per-call and reusable-buffer production
//! closures, timing host transfer, Hephaestus dispatch, and readback.
//! One largest representative per dimensional regime preserves the measured
//! preparation and dispatch behavior while keeping 100 observations per arm
//! within the suite's 300-second bound.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use apollo_nufft::{
    NufftGpuBuffers1D, NufftGpuBuffers3D, NufftWgpuBackend, NufftWgpuPlan1D, NufftWgpuPlan3D,
    UniformDomain1D, UniformGrid3D,
};
use eunomia::Complex32;
use leto::Array3;
use std::hint::black_box;

const ONE_DIMENSIONAL_CASES: [(usize, usize); 1] = [(256, 256)];
const THREE_DIMENSIONAL_CASES: [usize; 1] = [8];

fn positions(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (index as f32 / count as f32) * std::f32::consts::TAU)
        .collect()
}

fn values(count: usize) -> Vec<Complex32> {
    (0..count)
        .map(|index| {
            let t = index as f32 / count.max(1) as f32;
            Complex32::new(
                (t * std::f32::consts::TAU).cos(),
                (t * std::f32::consts::PI).sin(),
            )
        })
        .collect()
}

fn plan_1d(n: usize, oversampling: usize, kernel_width: usize) -> NufftWgpuPlan1D {
    let domain = UniformDomain1D::new(n, std::f64::consts::TAU / n as f64)
        .expect("invariant: benchmark domain parameters are valid");
    NufftWgpuPlan1D::new(domain, oversampling, kernel_width)
}

fn positions_3d(count: usize) -> Vec<(f32, f32, f32)> {
    let extent = std::f32::consts::TAU;
    let denominator = count.max(1);
    (0..count)
        .map(|index| {
            (
                (index as f32 / denominator as f32 * extent) % extent,
                ((index * 3 + 1) as f32 / denominator as f32 * extent) % extent,
                ((index * 7 + 2) as f32 / denominator as f32 * extent) % extent,
            )
        })
        .collect()
}

fn modes_3d(n: usize) -> Array3<Complex32> {
    Array3::from_shape_fn([n, n, n], |[ix, iy, iz]| {
        let phase = (ix as f32 * 0.1 + iy as f32 * 0.07 - iz as f32 * 0.13) * std::f32::consts::TAU;
        Complex32::new(phase.cos(), phase.sin())
    })
}

fn plan_3d(n: usize, oversampling: usize, kernel_width: usize) -> Option<NufftWgpuPlan3D> {
    let spacing = std::f64::consts::TAU / n as f64;
    let grid = UniformGrid3D::new(n, n, n, spacing, spacing, spacing).ok()?;
    Some(NufftWgpuPlan3D::new(grid, oversampling, kernel_width))
}

fn try_backend() -> Option<NufftWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
        "apollo-nufft-wgpu-bench",
        hephaestus_core::DevicePreference::HighPerformance,
        &[],
        NufftWgpuBackend::required_device_limits(),
    ) {
        Ok(device) => Some(NufftWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("NUFFT GPU benchmark requires a working provider: {error}"),
    }
}

fn bench_fast_type1_1d(suite: &mut BenchmarkSuite) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping fast_type1_1d buffer_reuse benchmarks");
        return;
    };

    for (n, sample_count) in ONE_DIMENSIONAL_CASES {
        let plan = plan_1d(n, 2, 4);
        let positions = positions(sample_count);
        let amplitudes = values(sample_count);

        suite.run(
            BenchmarkCase::new("nufft_fast_type1_1d", "per_call", n),
            || {
                let output = backend
                    .execute_fast_type1_1d(
                        black_box(&plan),
                        black_box(&positions),
                        black_box(&amplitudes),
                    )
                    .expect("fast type1 1d per-call");
                black_box(output);
            },
        );

        let mut buffers = NufftGpuBuffers1D::new(backend.device(), &plan, sample_count)
            .expect("provider buffer allocation");
        let mut output = vec![eunomia::Complex64::new(0.0, 0.0); n];
        suite.run(
            BenchmarkCase::new("nufft_fast_type1_1d", "with_buffers", n),
            || {
                backend
                    .execute_fast_type1_1d_with_buffers(
                        black_box(&mut buffers),
                        black_box(&positions),
                        black_box(&amplitudes),
                        black_box(&mut output),
                    )
                    .expect("fast type1 1d with_buffers");
                black_box(&output);
            },
        );
    }
}

fn bench_fast_type2_1d(suite: &mut BenchmarkSuite) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping fast_type2_1d buffer_reuse benchmarks");
        return;
    };

    for (n, sample_count) in ONE_DIMENSIONAL_CASES {
        let plan = plan_1d(n, 2, 4);
        let coefficients = values(n);
        let positions = positions(sample_count);

        suite.run(
            BenchmarkCase::new("nufft_fast_type2_1d", "per_call", n),
            || {
                let output = backend
                    .execute_fast_type2_1d(
                        black_box(&plan),
                        black_box(&coefficients),
                        black_box(&positions),
                    )
                    .expect("fast type2 1d per-call");
                black_box(output);
            },
        );

        let mut buffers = NufftGpuBuffers1D::new(backend.device(), &plan, sample_count)
            .expect("provider buffer allocation");
        let mut output = vec![eunomia::Complex64::new(0.0, 0.0); sample_count];
        suite.run(
            BenchmarkCase::new("nufft_fast_type2_1d", "with_buffers", n),
            || {
                backend
                    .execute_fast_type2_1d_with_buffers(
                        black_box(&mut buffers),
                        black_box(&coefficients),
                        black_box(&positions),
                        black_box(&mut output),
                    )
                    .expect("fast type2 1d with_buffers");
                black_box(&output);
            },
        );
    }
}

fn bench_fast_type1_3d(suite: &mut BenchmarkSuite) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping fast_type1_3d buffer_reuse benchmarks");
        return;
    };

    for n in THREE_DIMENSIONAL_CASES {
        let Some(plan) = plan_3d(n, 2, 4) else {
            eprintln!("fast_type1_3d: skipping n={n} (plan construction failed)");
            continue;
        };
        let sample_count = n;
        let positions = positions_3d(sample_count);
        let amplitudes = values(sample_count);

        suite.run(
            BenchmarkCase::new("nufft_fast_type1_3d", "per_call", n),
            || {
                let output = backend
                    .execute_fast_type1_3d(
                        black_box(&plan),
                        black_box(&positions),
                        black_box(&amplitudes),
                    )
                    .expect("fast type1 3d per-call");
                black_box(output);
            },
        );

        let mut buffers = NufftGpuBuffers3D::new(backend.device(), &plan, sample_count)
            .expect("provider buffer allocation");
        let mut output = vec![eunomia::Complex64::new(0.0, 0.0); n * n * n];
        suite.run(
            BenchmarkCase::new("nufft_fast_type1_3d", "with_buffers", n),
            || {
                backend
                    .execute_fast_type1_3d_with_buffers(
                        black_box(&mut buffers),
                        black_box(&positions),
                        black_box(&amplitudes),
                        black_box(&mut output),
                    )
                    .expect("fast type1 3d with_buffers");
                black_box(&output);
            },
        );
    }
}

fn bench_fast_type2_3d(suite: &mut BenchmarkSuite) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping fast_type2_3d buffer_reuse benchmarks");
        return;
    };

    for n in THREE_DIMENSIONAL_CASES {
        let Some(plan) = plan_3d(n, 2, 4) else {
            eprintln!("fast_type2_3d: skipping n={n} (plan construction failed)");
            continue;
        };
        let sample_count = n;
        let modes = modes_3d(n);
        let positions = positions_3d(sample_count);

        suite.run(
            BenchmarkCase::new("nufft_fast_type2_3d", "per_call", n),
            || {
                let output = backend
                    .execute_fast_type2_3d(
                        black_box(&plan),
                        black_box(&modes),
                        black_box(&positions),
                    )
                    .expect("fast type2 3d per-call");
                black_box(output);
            },
        );

        let mut buffers = NufftGpuBuffers3D::new(backend.device(), &plan, sample_count)
            .expect("provider buffer allocation");
        let mut output = vec![eunomia::Complex64::new(0.0, 0.0); sample_count];
        suite.run(
            BenchmarkCase::new("nufft_fast_type2_3d", "with_buffers", n),
            || {
                backend
                    .execute_fast_type2_3d_with_buffers(
                        black_box(&mut buffers),
                        black_box(&modes),
                        black_box(&positions),
                        black_box(&mut output),
                    )
                    .expect("fast type2 3d with_buffers");
                black_box(&output);
            },
        );
    }
}

fn main() -> Result<(), apollo_bench::BenchmarkModeError> {
    let mode = BenchmarkMode::from_environment()?;
    let mut suite = BenchmarkSuite::new(mode.apply(BenchmarkConfig::standard()));
    bench_fast_type1_1d(&mut suite);
    bench_fast_type2_1d(&mut suite);
    bench_fast_type1_3d(&mut suite);
    bench_fast_type2_3d(&mut suite);
    print!("{}", suite.report());
    Ok(())
}
