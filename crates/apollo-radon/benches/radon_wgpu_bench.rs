//! Criterion benchmarks for `apollo-radon-wgpu` forward projection and FBP.
//!
//! Measures wall-clock cost (GPU dispatch + PCIe readback) for:
//!
//! - [`bench_forward`]:              image → sinogram  (parallel-beam Radon projection)
//! - [`bench_filtered_backproject`]: sinogram → image  (Ram-Lak filter + adjoint backprojection)
//!
//! across three image sizes: 64×64, 128×128, and 256×256.
//!
//! # Mathematical justification
//!
//! **Forward projection** (parallel-beam Radon transform, Natterer 2001 §I.1):
//!
//! ```text
//! (Rf)(θ, s) = ∫∫ f(x, y) δ(x·cos θ + y·sin θ − s) dx dy
//! ```
//!
//! Discretised as a sum of bilinear interpolation weights over pixels at
//! each detector bin and angle.
//!
//! **Filtered backprojection** (FBP, Bracewell & Riddle 1967; Natterer 2001 §II.2):
//!
//! ```text
//! f̂(x, y) ≈ (π / N_θ) · Σ_{k=0}^{N_θ−1} (h * p_k)(x·cos θ_k + y·sin θ_k)
//! ```
//!
//! where `h` is the Ram-Lak ramp filter `ĥ[k] = 2π|k| / (N_d · Δ)`.
//!
//! **Test input**: Gaussian disk phantom
//!
//! ```text
//! f(x, y) = exp(−(x² + y²) / (2σ²)),  σ = 0.25
//! ```
//!
//! with pixel-centred coordinates normalised to `[−1, 1]`. This phantom has a
//! known analytical Radon transform `(Rf)(θ, s) = σ√(2π) · exp(−s²/(2σ²))`,
//! providing non-trivial frequency content on all axes and a stable workload.
//!
//! **Projection angles**: `θ_k = k·π / N_θ`, uniformly spaced in `[0, π)`.
//!
//! **Detector spacing**: `Δ = 2√2 / (N_d − 1)` for an `N_d`-bin array whose
//! aperture `[−√2, √2]` fully subtends the image diagonal at all angles.
//!
//! **Parameter sets** `(image_size, angle_count, detector_count)`:
//!
//! | image_size | angle_count | detector_count | rationale |
//! |---|---|---|---|
//! | 64  | 90  | 91  | ⌈64·√2⌉ + 1 bins, ~90 angles covers [0,π) at 1°/step |
//! | 128 | 180 | 182 | ⌈128·√2⌉ + 1 bins, ~180 angles at 1°/step |
//! | 256 | 360 | 362 | ⌈256·√2⌉ + 1 bins, ~360 angles at 0.5°/step |

#![allow(missing_docs)]

use apollo_radon::{RadonWgpuBackend, RadonWgpuPlan};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use leto::Array2;

/// Parameter sets: `(image_size, angle_count, detector_count)`.
///
/// - `image_size`:      side length of the square image (rows == cols).
/// - `angle_count`:     equally-spaced angles in `[0, π)`.
/// - `detector_count`:  bin count; `⌈image_size · √2⌉ + 1` fully subtends the diagonal.
const PARAMS: &[(usize, usize, usize)] = &[(64, 90, 91), (128, 180, 182), (256, 360, 362)];

/// Gaussian disk phantom with pixel-centred coordinates normalised to `[−1, 1]`.
///
/// `f(x, y) = exp(−(x² + y²) / (2σ²))`, σ = 0.25.
///
/// Analytical Radon transform: `(Rf)(θ, s) = σ√(2π) · exp(−s² / (2σ²))`,
/// independent of θ due to circular symmetry. Provides a stable, non-trivial
/// benchmark workload with known spectral content.
fn gaussian_phantom(rows: usize, cols: usize) -> Array2<f32> {
    const SIGMA_SQ: f32 = 0.25 * 0.25;
    Array2::from_shape_fn([rows, cols], |[i, j]| {
        let x = 2.0 * i as f32 / (rows - 1) as f32 - 1.0;
        let y = 2.0 * j as f32 / (cols - 1) as f32 - 1.0;
        (-(x * x + y * y) / (2.0 * SIGMA_SQ)).exp()
    })
}

/// Projection angles `θ_k = k·π / angle_count`, uniformly spaced in `[0, π)`.
fn projection_angles(angle_count: usize) -> Vec<f32> {
    (0..angle_count)
        .map(|k| std::f32::consts::PI * k as f32 / angle_count as f32)
        .collect()
}

/// Detector spacing `Δ = 2√2 / (detector_count − 1)`.
///
/// Ensures the full image diagonal (length `2√2` for a `[−1, 1]²` domain) is
/// subtended at all projection angles.
fn detector_spacing(detector_count: usize) -> f64 {
    2.0 * std::f64::consts::SQRT_2 / (detector_count - 1) as f64
}

/// Acquire a `RadonWgpuBackend`, or return `None` when no device is available.
fn try_backend() -> Option<RadonWgpuBackend> {
    RadonWgpuBackend::try_default().ok()
}

/// Benchmark GPU forward parallel-beam Radon projection across three image sizes.
///
/// Measures end-to-end cost: host→GPU upload, projection compute (one thread group
/// per output pixel), GPU→host readback via staging buffer.
///
/// Input: Gaussian disk phantom (non-trivial, analytically characterisable).
/// Output: sinogram of shape `(angle_count, detector_count)`.
fn bench_forward(c: &mut Criterion) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping bench_forward");
        return;
    };

    let mut group = c.benchmark_group("radon_wgpu_forward");

    for &(image_size, angle_count, detector_count) in PARAMS {
        let plan = RadonWgpuPlan::new(
            image_size,
            image_size,
            angle_count,
            detector_count,
            detector_spacing(detector_count).to_bits(),
        );
        let image = gaussian_phantom(image_size, image_size);
        let angles = projection_angles(angle_count);

        group.bench_with_input(
            BenchmarkId::new("image_size", image_size),
            &image_size,
            |b, _| {
                b.iter(|| {
                    backend
                        .execute_forward(black_box(&plan), black_box(&image), black_box(&angles))
                        .expect("GPU forward Radon projection")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GPU ramp-filtered backprojection (FBP) across three image sizes.
///
/// Two-pass GPU execution measured per iteration:
/// 1. Ram-Lak ramp filter applied row-wise to the sinogram
///    (`ĥ[k] = 2π|k|/(N_d·Δ)`; circular convolution via FFT, Bracewell & Riddle 1967).
/// 2. Adjoint backprojection of the filtered sinogram (Natterer 2001 §II.2),
///    followed by `π / angle_count` normalisation.
///
/// The sinogram is pre-computed once via `execute_forward` outside the measurement
/// loop so only the FBP dispatch (filter + backprojection + readback) is timed.
fn bench_filtered_backproject(c: &mut Criterion) {
    let Some(backend) = try_backend() else {
        eprintln!("No WGPU device available; skipping bench_filtered_backproject");
        return;
    };

    let mut group = c.benchmark_group("radon_wgpu_fbp");

    for &(image_size, angle_count, detector_count) in PARAMS {
        let plan = RadonWgpuPlan::new(
            image_size,
            image_size,
            angle_count,
            detector_count,
            detector_spacing(detector_count).to_bits(),
        );
        let image = gaussian_phantom(image_size, image_size);
        let angles = projection_angles(angle_count);

        // Pre-compute sinogram once; only FBP execution is measured in the loop.
        let sinogram = backend
            .execute_forward(&plan, &image, &angles)
            .expect("forward pass for FBP benchmark setup");

        group.bench_with_input(
            BenchmarkId::new("image_size", image_size),
            &image_size,
            |b, _| {
                b.iter(|| {
                    backend
                        .execute_filtered_backproject(
                            black_box(&plan),
                            black_box(&sinogram),
                            black_box(&angles),
                        )
                        .expect("GPU filtered backprojection")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_forward, bench_filtered_backproject);
criterion_main!(benches);
