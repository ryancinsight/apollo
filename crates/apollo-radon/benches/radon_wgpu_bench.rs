//! Native Apollo benchmarks for provider-backed Radon projection and FBP.
//!
//! The Gaussian phantom has analytical forward transform
//! `σ√(2π) exp(-s²/(2σ²))`; each measured closure retains the original
//! host-to-device upload, Hephaestus execution, and readback workload.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkSuite};
use apollo_radon::{RadonWgpuBackend, RadonWgpuPlan};
use leto::Array2;
use std::hint::black_box;

const PARAMETERS: &[(usize, usize, usize)] = &[(64, 90, 91), (128, 180, 182), (256, 360, 362)];

fn gaussian_phantom(rows: usize, columns: usize) -> Array2<f32> {
    const SIGMA_SQUARED: f32 = 0.25 * 0.25;
    Array2::from_shape_fn([rows, columns], |[row, column]| {
        let x = 2.0 * row as f32 / (rows - 1) as f32 - 1.0;
        let y = 2.0 * column as f32 / (columns - 1) as f32 - 1.0;
        (-(x * x + y * y) / (2.0 * SIGMA_SQUARED)).exp()
    })
}

fn projection_angles(angle_count: usize) -> Vec<f32> {
    (0..angle_count)
        .map(|index| std::f32::consts::PI * index as f32 / angle_count as f32)
        .collect()
}

fn detector_spacing(detector_count: usize) -> f64 {
    2.0 * std::f64::consts::SQRT_2 / (detector_count - 1) as f64
}

fn bench_forward(suite: &mut BenchmarkSuite) {
    let Ok(backend) = RadonWgpuBackend::try_default() else {
        eprintln!("No WGPU device available; skipping Radon forward benchmarks");
        return;
    };

    for &(image_size, angle_count, detector_count) in PARAMETERS {
        let plan = RadonWgpuPlan::new(
            image_size,
            image_size,
            angle_count,
            detector_count,
            detector_spacing(detector_count).to_bits(),
        );
        let image = gaussian_phantom(image_size, image_size);
        let angles = projection_angles(angle_count);
        suite.run(
            BenchmarkCase::new("radon_wgpu_forward", "image_size", image_size),
            || {
                let sinogram = backend
                    .execute_forward(black_box(&plan), black_box(&image), black_box(&angles))
                    .expect("GPU forward Radon projection");
                black_box(sinogram);
            },
        );
    }
}

fn bench_filtered_backproject(suite: &mut BenchmarkSuite) {
    let Ok(backend) = RadonWgpuBackend::try_default() else {
        eprintln!("No WGPU device available; skipping Radon FBP benchmarks");
        return;
    };

    for &(image_size, angle_count, detector_count) in PARAMETERS {
        let plan = RadonWgpuPlan::new(
            image_size,
            image_size,
            angle_count,
            detector_count,
            detector_spacing(detector_count).to_bits(),
        );
        let image = gaussian_phantom(image_size, image_size);
        let angles = projection_angles(angle_count);
        let sinogram = backend
            .execute_forward(&plan, &image, &angles)
            .expect("forward pass for FBP benchmark setup");

        suite.run(
            BenchmarkCase::new("radon_wgpu_fbp", "image_size", image_size),
            || {
                let reconstruction = backend
                    .execute_filtered_backproject(
                        black_box(&plan),
                        black_box(&sinogram),
                        black_box(&angles),
                    )
                    .expect("GPU filtered backprojection");
                black_box(reconstruction);
            },
        );
    }
}

fn main() {
    let mut suite = BenchmarkSuite::default();
    bench_forward(&mut suite);
    bench_filtered_backproject(&mut suite);
    suite.emit();
}
