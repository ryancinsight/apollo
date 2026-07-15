//! Native Apollo benchmarks for FFT Hephaestus buffer reuse.
//!
//! Measures end-to-end host transfer, provider dispatch, and readback with
//! either per-call allocation or reusable typed buffers.

#![allow(missing_docs)]

use apollo_bench::{BenchmarkCase, BenchmarkSuite};
use apollo_fft::{GpuFft3d, GpuFft3dBuffers};
use leto::Array3;
use std::hint::black_box;

fn real_field(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
    Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| {
        let value = (i + j + k) as f64;
        (0.057 * value).sin() + 0.3 * (0.11 * value).cos()
    })
}

fn try_fft_plan(nx: usize, ny: usize, nz: usize) -> Option<GpuFft3d> {
    let device = hephaestus_wgpu::WgpuDevice::try_default("apollo-fft-wgpu-bench").ok()?;
    GpuFft3d::new(device, nx, ny, nz).ok()
}

fn bench_forward_3d(suite: &mut BenchmarkSuite) {
    for (nx, ny, nz) in [(4_usize, 4, 4), (8, 8, 8), (16, 16, 16)] {
        let Some(plan) = try_fft_plan(nx, ny, nz) else {
            eprintln!("No WGPU device; skipping bench_forward_3d n={nx}");
            return;
        };
        let field = real_field(nx, ny, nz);
        let mut output = vec![0.0_f32; 2 * nx * ny * nz];

        suite.run(
            BenchmarkCase::new("fft3d_forward", "allocating", nx),
            || {
                let result = plan
                    .forward(black_box(&field))
                    .expect("GPU readback must succeed");
                black_box(result);
            },
        );

        let mut buffers = GpuFft3dBuffers::new(&plan).expect("provider buffer allocation");
        suite.run(
            BenchmarkCase::new("fft3d_forward", "with_buffers", nx),
            || {
                plan.forward_into_with_buffers(
                    black_box(&field),
                    black_box(&mut output),
                    black_box(&mut buffers),
                )
                .expect("GPU readback must succeed");
                black_box(&output);
            },
        );
    }
}

fn bench_inverse_3d(suite: &mut BenchmarkSuite) {
    for (nx, ny, nz) in [(4_usize, 4, 4), (8, 8, 8), (16, 16, 16)] {
        let Some(plan) = try_fft_plan(nx, ny, nz) else {
            eprintln!("No WGPU device; skipping bench_inverse_3d n={nx}");
            return;
        };
        let field = real_field(nx, ny, nz);
        let spectrum = plan.forward(&field).expect("GPU readback must succeed");
        let mut allocating_output = Array3::<f64>::zeros([nx, ny, nz]);

        suite.run(
            BenchmarkCase::new("fft3d_inverse", "allocating", nx),
            || {
                plan.inverse(black_box(&spectrum), &mut allocating_output)
                    .expect("GPU readback must succeed");
                black_box(&allocating_output);
            },
        );

        let mut buffers = GpuFft3dBuffers::new(&plan).expect("provider buffer allocation");
        let mut buffered_output = Array3::<f64>::zeros([nx, ny, nz]);
        suite.run(
            BenchmarkCase::new("fft3d_inverse", "with_buffers", nx),
            || {
                plan.inverse_with_buffers(
                    black_box(&spectrum),
                    &mut buffered_output,
                    black_box(&mut buffers),
                )
                .expect("GPU readback must succeed");
                black_box(&buffered_output);
            },
        );
    }
}

fn main() {
    let mut suite = BenchmarkSuite::default();
    bench_forward_3d(&mut suite);
    bench_inverse_3d(&mut suite);
    suite.emit();
}
