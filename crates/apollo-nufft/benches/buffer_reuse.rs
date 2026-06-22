//! Criterion benchmarks for NUFFT WGPU fast-path buffer reuse.
//!
//! These benchmarks measure the wall-clock cost (GPU dispatch + PCIe readback)
//! of repeated fast-path NUFFT calls with and without pre-allocated GPU buffers.
//!
//! # Mathematical justification
//!
//! Per-call cost: device buffer allocation + bind group creation + upload + dispatch + readback.
//! With-buffers cost: upload + bind group creation + dispatch + readback (buffers pre-allocated).
//! Expected speedup for small N: significant (allocation dominates); for large N: proportional
//! to compute, not allocation.
//!
//! ## 3D oversampled dimension
//!
//! For oversampling σ and kernel half-width w, the radix-2 oversampled length per axis is
//! `max(n·σ, 2w+1).next_power_of_two()`. With σ=2, w=4 and n∈{4,6,8} this evaluates to 16
//! for all three values, so a single `oversampled_3d_size` helper covers the bench matrix.

#![allow(missing_docs)]

use apollo_nufft::{
    NufftGpuBuffers1D, NufftGpuBuffers3D, NufftWgpuBackend, NufftWgpuPlan1D, NufftWgpuPlan3D,
    UniformDomain1D, UniformGrid3D,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array3;
use num_complex::Complex32;

// ---------------------------------------------------------------------------
// 1D helpers (unchanged from original)
// ---------------------------------------------------------------------------

/// Deterministic non-uniform positions in `[0, 2π)`.
fn positions(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (i as f32 / count as f32) * std::f32::consts::TAU)
        .collect()
}

/// Deterministic complex amplitudes.
fn values(count: usize) -> Vec<Complex32> {
    (0..count)
        .map(|i| {
            let t = i as f32 / count.max(1) as f32;
            Complex32::new(
                (t * std::f32::consts::TAU).cos(),
                (t * std::f32::consts::PI).sin(),
            )
        })
        .collect()
}

/// Build a valid 1D NUFFT WGPU plan for the given parameters.
fn plan_1d(n: usize, oversampling: usize, kernel_width: usize) -> NufftWgpuPlan1D {
    let domain = UniformDomain1D::new(n, std::f64::consts::TAU / n as f64)
        .expect("bench domain parameters are valid");
    NufftWgpuPlan1D::new(domain, oversampling, kernel_width)
}

// ---------------------------------------------------------------------------
// 3D helpers
// ---------------------------------------------------------------------------

/// Compute the radix-2 oversampled axis length used by the 3D fast path.
///
/// Mirrors `supported_radix2_oversampled_len` from `apollo-nufft`:
/// `max(n·sigma, 2·kernel_width + 1).next_power_of_two()`.
fn oversampled_3d_size(n: usize, oversampling: usize, kernel_width: usize) -> usize {
    let lower = n
        .checked_mul(oversampling)
        .expect("oversampled length overflow")
        .max(2 * kernel_width + 1);
    lower.next_power_of_two()
}

/// Deterministic 3D non-uniform positions in `[0, 2π)³`.
///
/// Coprime stride multipliers (1, 3, 7) decorrelate the three axes while keeping
/// all coordinates within `[0, L)` after modular reduction with `L = 2π`.
fn positions_3d(count: usize) -> Vec<(f32, f32, f32)> {
    let l = std::f32::consts::TAU;
    let m = count.max(1);
    (0..count)
        .map(|i| {
            (
                (i as f32 / m as f32 * l) % l,
                ((i * 3 + 1) as f32 / m as f32 * l) % l,
                ((i * 7 + 2) as f32 / m as f32 * l) % l,
            )
        })
        .collect()
}

/// Deterministic complex-valued mode array of shape `(n, n, n)`.
///
/// Values are derived from a linear phase: `exp(i·2π·(0.1·kx + 0.07·ky − 0.13·kz))`,
/// which provides non-trivial coverage across all mode indices.
fn modes_3d(n: usize) -> Array3<Complex32> {
    Array3::from_shape_fn((n, n, n), |(ix, iy, iz)| {
        let phase = (ix as f32 * 0.1 + iy as f32 * 0.07 - iz as f32 * 0.13) * std::f32::consts::TAU;
        Complex32::new(phase.cos(), phase.sin())
    })
}

/// Build a valid 3D NUFFT WGPU plan.
///
/// Returns `None` if `UniformGrid3D::new` rejects the parameters (e.g., `n = 0`).
fn plan_3d(n: usize, oversampling: usize, kernel_width: usize) -> Option<NufftWgpuPlan3D> {
    let dx = std::f64::consts::TAU / n as f64;
    let grid = UniformGrid3D::new(n, n, n, dx, dx, dx).ok()?;
    Some(NufftWgpuPlan3D::new(grid, oversampling, kernel_width))
}

// ---------------------------------------------------------------------------
// 1D benchmarks (unchanged from original)
// ---------------------------------------------------------------------------

/// Benchmark per-call vs reusable-buffer fast Type-1 1D NUFFT.
fn bench_fast_type1_1d(c: &mut Criterion) {
    let Ok(backend) = NufftWgpuBackend::try_default() else {
        eprintln!("No WGPU device available; skipping fast_type1_1d buffer_reuse benchmarks");
        return;
    };

    let oversampling = 2;
    let kernel_width = 4;

    for (n, m_count) in [(64usize, 64usize), (128, 128), (256, 256)] {
        let plan = plan_1d(n, oversampling, kernel_width);
        let pos = positions(m_count);
        let vals = values(m_count);
        let oversampled_len = n * oversampling;

        let mut group = c.benchmark_group(format!("nufft_fast_type1_1d_n{n}_m{m_count}"));

        group.bench_function(BenchmarkId::new("per_call", n), |b| {
            b.iter(|| {
                let out = backend
                    .execute_fast_type1_1d(black_box(&plan), black_box(&pos), black_box(&vals))
                    .expect("fast type1 1d per-call");
                black_box(out);
            });
        });

        let buffers = NufftGpuBuffers1D::new(backend.device(), n, oversampled_len, m_count);
        group.bench_function(BenchmarkId::new("with_buffers", n), |b| {
            b.iter(|| {
                let out = backend
                    .execute_fast_type1_1d_with_buffers(
                        black_box(&plan),
                        black_box(&buffers),
                        black_box(&pos),
                        black_box(&vals),
                    )
                    .expect("fast type1 1d with_buffers");
                black_box(out);
            });
        });

        group.finish();
    }
}

/// Benchmark per-call vs reusable-buffer fast Type-2 1D NUFFT.
fn bench_fast_type2_1d(c: &mut Criterion) {
    let Ok(backend) = NufftWgpuBackend::try_default() else {
        eprintln!("No WGPU device available; skipping fast_type2_1d buffer_reuse benchmarks");
        return;
    };

    let oversampling = 2;
    let kernel_width = 4;

    for (n, m_count) in [(64usize, 64usize), (128, 128), (256, 256)] {
        let plan = plan_1d(n, oversampling, kernel_width);
        let coeffs: Vec<Complex32> = values(n);
        let pos = positions(m_count);
        let oversampled_len = n * oversampling;

        let mut group = c.benchmark_group(format!("nufft_fast_type2_1d_n{n}_m{m_count}"));

        group.bench_function(BenchmarkId::new("per_call", n), |b| {
            b.iter(|| {
                let out = backend
                    .execute_fast_type2_1d(black_box(&plan), black_box(&coeffs), black_box(&pos))
                    .expect("fast type2 1d per-call");
                black_box(out);
            });
        });

        let buffers = NufftGpuBuffers1D::new(backend.device(), n, oversampled_len, m_count);
        group.bench_function(BenchmarkId::new("with_buffers", n), |b| {
            b.iter(|| {
                let out = backend
                    .execute_fast_type2_1d_with_buffers(
                        black_box(&plan),
                        black_box(&buffers),
                        black_box(&coeffs),
                        black_box(&pos),
                    )
                    .expect("fast type2 1d with_buffers");
                black_box(out);
            });
        });

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// 3D benchmarks
// ---------------------------------------------------------------------------

/// Benchmark per-call vs reusable-buffer fast Type-1 3D NUFFT.
///
/// Uses `oversampling = 2`, `kernel_width = 4`, `n ∈ {4, 6, 8}`, `M = n`.
/// For each `n`, `oversampled_3d_size(n, 2, 4) = 16`, so the pre-allocated
/// `NufftGpuBuffers3D` covers the full oversampled 16³ grid.
///
/// `n = 6` exercises the non-power-of-two Bluestein/radix-2-padded code path;
/// it is skipped silently if plan construction fails.
fn bench_fast_type1_3d(c: &mut Criterion) {
    let Ok(backend) = NufftWgpuBackend::try_default() else {
        eprintln!("No WGPU device available; skipping fast_type1_3d buffer_reuse benchmarks");
        return;
    };

    let oversampling = 2usize;
    let kernel_width = 4usize;

    for n in [4usize, 6, 8] {
        let Some(plan) = plan_3d(n, oversampling, kernel_width) else {
            eprintln!("fast_type1_3d: skipping n={n} (plan construction failed)");
            continue;
        };

        let m_count = n;
        let pos = positions_3d(m_count);
        let vals = values(m_count);
        let oversampled_n = oversampled_3d_size(n, oversampling, kernel_width);
        let shape = (n, n, n);
        let oversampled = (oversampled_n, oversampled_n, oversampled_n);

        let mut group = c.benchmark_group(format!("nufft_fast_type1_3d_n{n}_m{m_count}"));

        group.bench_function(BenchmarkId::new("per_call", n), |b| {
            b.iter(|| {
                let out = backend
                    .execute_fast_type1_3d(black_box(&plan), black_box(&pos), black_box(&vals))
                    .expect("fast type1 3d per-call");
                black_box(out);
            });
        });

        let buffers = NufftGpuBuffers3D::new(backend.device(), shape, oversampled, m_count);
        group.bench_function(BenchmarkId::new("with_buffers", n), |b| {
            b.iter(|| {
                let out = backend
                    .execute_fast_type1_3d_with_buffers(
                        black_box(&plan),
                        black_box(&buffers),
                        black_box(&pos),
                        black_box(&vals),
                    )
                    .expect("fast type1 3d with_buffers");
                black_box(out);
            });
        });

        group.finish();
    }
}

/// Benchmark per-call vs reusable-buffer fast Type-2 3D NUFFT.
///
/// Uses `oversampling = 2`, `kernel_width = 4`, `n ∈ {4, 6, 8}`, `M = n`.
/// Mode array has shape `(n, n, n)` with analytically derived phase values.
/// `n = 6` exercises the non-power-of-two padded code path; skipped if construction fails.
fn bench_fast_type2_3d(c: &mut Criterion) {
    let Ok(backend) = NufftWgpuBackend::try_default() else {
        eprintln!("No WGPU device available; skipping fast_type2_3d buffer_reuse benchmarks");
        return;
    };

    let oversampling = 2usize;
    let kernel_width = 4usize;

    for n in [4usize, 6, 8] {
        let Some(plan) = plan_3d(n, oversampling, kernel_width) else {
            eprintln!("fast_type2_3d: skipping n={n} (plan construction failed)");
            continue;
        };

        let m_count = n;
        let modes = modes_3d(n);
        let pos = positions_3d(m_count);
        let oversampled_n = oversampled_3d_size(n, oversampling, kernel_width);
        let shape = (n, n, n);
        let oversampled = (oversampled_n, oversampled_n, oversampled_n);

        let mut group = c.benchmark_group(format!("nufft_fast_type2_3d_n{n}_m{m_count}"));

        group.bench_function(BenchmarkId::new("per_call", n), |b| {
            b.iter(|| {
                let out = backend
                    .execute_fast_type2_3d(black_box(&plan), black_box(&modes), black_box(&pos))
                    .expect("fast type2 3d per-call");
                black_box(out);
            });
        });

        let buffers = NufftGpuBuffers3D::new(backend.device(), shape, oversampled, m_count);
        group.bench_function(BenchmarkId::new("with_buffers", n), |b| {
            b.iter(|| {
                let out = backend
                    .execute_fast_type2_3d_with_buffers(
                        black_box(&plan),
                        black_box(&buffers),
                        black_box(&modes),
                        black_box(&pos),
                    )
                    .expect("fast type2 3d with_buffers");
                black_box(out);
            });
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_fast_type1_1d,
    bench_fast_type2_1d,
    bench_fast_type1_3d,
    bench_fast_type2_3d
);
criterion_main!(benches);
