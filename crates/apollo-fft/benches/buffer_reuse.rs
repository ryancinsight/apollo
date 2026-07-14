//! Criterion benchmarks for FFT-WGPU 3D buffer reuse.
//!
//! Measures the wall-clock cost (GPU dispatch + PCIe readback) of repeated
//! 3D FFT calls with and without pre-allocated GPU buffers, across several
//! grid sizes.
//!
//! # Mathematical justification
//!
//! Parseval's theorem: the 3D FFT preserves energy (‖F‖² = N·‖f‖²).
//! Benchmark measures total round-trip cost: host→GPU upload, compute, GPU→host
//! readback. Buffer reuse eliminates `wgpu::Device::create_buffer` calls on
//! every dispatch, which are O(1) but involve driver allocation and zeroing.

#![allow(missing_docs)]

use apollo_fft::{GpuFft3d, GpuFft3dBuffers};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use leto::Array3;
use std::hint::black_box;

/// Build a deterministic real test field of shape `(nx, ny, nz)`.
///
/// Values are drawn from the analytic signal `sin(0.057·x) + 0.3·cos(0.11·x)`
/// where `x = i + j + k`, ensuring non-trivial frequency content on all axes.
fn real_field(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
    Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| {
        let x = (i + j + k) as f64;
        (0.057 * x).sin() + 0.3 * (0.11 * x).cos()
    })
}

/// Acquire a WGPU device and build a `GpuFft3d` plan, or return `None` if unavailable.
fn try_fft_plan(nx: usize, ny: usize, nz: usize) -> Option<GpuFft3d> {
    let device = apollo_wgpu_helpers::WgpuDevice::try_default("apollo-fft-wgpu-bench").ok()?;
    GpuFft3d::new(device.device().clone(), device.queue().clone(), nx, ny, nz).ok()
}

/// Benchmark the forward 3D FFT comparing per-call-allocating vs buffer-reuse paths.
///
/// "allocating": `GpuFft3d::forward` — allocates a `Vec<f32>` and internal
///   scratch vectors on every call.
///
/// "with_buffers": `GpuFft3d::forward_into_with_buffers` — reuses pre-allocated
///   GPU buffers and host scratch, eliminating driver allocation overhead.
fn bench_forward_3d(c: &mut Criterion) {
    for (nx, ny, nz) in [(4usize, 4, 4), (8, 8, 8), (16, 16, 16)] {
        let Some(plan) = try_fft_plan(nx, ny, nz) else {
            eprintln!("No WGPU device; skipping bench_forward_3d n={nx}");
            return;
        };
        let field = real_field(nx, ny, nz);
        let n = nx * ny * nz;
        let mut out = vec![0.0_f32; 2 * n];

        let mut group = c.benchmark_group(format!("fft3d_forward_nx{nx}"));

        // Allocating path: `forward` allocates Vec<f32> internally per call.
        group.bench_function(BenchmarkId::new("allocating", nx), |b| {
            b.iter(|| {
                let result = plan
                    .forward(black_box(&field))
                    .expect("GPU readback must succeed");
                black_box(result);
            });
        });

        // Reuse path: caller retains GPU buffers across repeated calls.
        let mut buffers = GpuFft3dBuffers::new(&plan);
        group.bench_function(BenchmarkId::new("with_buffers", nx), |b| {
            b.iter(|| {
                plan.forward_into_with_buffers(
                    black_box(&field),
                    black_box(&mut out),
                    black_box(&mut buffers),
                )
                .expect("GPU readback must succeed");
                black_box(&out);
            });
        });

        group.finish();
    }
}

/// Benchmark the inverse 3D FFT comparing per-call-allocating vs buffer-reuse paths.
///
/// "allocating": `GpuFft3d::inverse` — allocates temporary `Vec<f32>` scratch
///   vectors per call.
///
/// "with_buffers": `GpuFft3d::inverse_with_buffers` — reuses pre-allocated
///   GPU buffers, eliminating per-call driver allocation.
fn bench_inverse_3d(c: &mut Criterion) {
    for (nx, ny, nz) in [(4usize, 4, 4), (8, 8, 8), (16, 16, 16)] {
        let Some(plan) = try_fft_plan(nx, ny, nz) else {
            eprintln!("No WGPU device; skipping bench_inverse_3d n={nx}");
            return;
        };
        let field = real_field(nx, ny, nz);
        let spectrum = plan.forward(&field).expect("GPU readback must succeed");

        let mut group = c.benchmark_group(format!("fft3d_inverse_nx{nx}"));

        // Allocating path: `inverse` allocates scratch vecs internally per call.
        group.bench_function(BenchmarkId::new("allocating", nx), |b| {
            let mut out = Array3::<f64>::zeros([nx, ny, nz]);
            b.iter(|| {
                plan.inverse(black_box(&spectrum), &mut out)
                    .expect("GPU readback must succeed");
                black_box(&out);
            });
        });

        // Reuse path: caller retains GPU buffers across repeated calls.
        let mut buffers = GpuFft3dBuffers::new(&plan);
        group.bench_function(BenchmarkId::new("with_buffers", nx), |b| {
            let mut out = Array3::<f64>::zeros([nx, ny, nz]);
            b.iter(|| {
                plan.inverse_with_buffers(black_box(&spectrum), &mut out, black_box(&mut buffers))
                    .expect("GPU readback must succeed");
                black_box(&out);
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_forward_3d, bench_inverse_3d);
criterion_main!(benches);
