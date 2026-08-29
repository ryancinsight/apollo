//! Reusable-buffer capacity and value-equivalence contracts.

use eunomia::Complex32;
use leto::{Array3, Layout, VecStorage};

use crate::{
    infrastructure::transport::gpu::{
        NufftGpuBuffers1D, NufftGpuBuffers3D, NufftWgpuPlan1D, NufftWgpuPlan3D,
    },
    UniformDomain1D, UniformGrid3D,
};

use super::support::{
    assert_input_length_mismatch, backend, grid3d, mode_components3d, modes3d, positions3d,
};

fn shifted_mode(kx: usize, ky: usize, kz: usize, shift: f32) -> Complex32 {
    let (re, im) = mode_components3d(kx, ky, kz);
    Complex32::new(re + shift, im - 0.5 * shift)
}

fn shifted_modes3d(shift: f32) -> Array3<Complex32> {
    Array3::from_shape_fn([3, 2, 2], |[kx, ky, kz]| shifted_mode(kx, ky, kz, shift))
}

fn strided_shifted_modes3d(shift: f32) -> Array3<Complex32> {
    let layout = Layout::try_new([3, 2, 2], [1, 6, 3], 0).expect("strided mode layout");
    let mut storage = vec![Complex32::new(0.0, 0.0); 12];
    for kx in 0..3 {
        for ky in 0..2 {
            for kz in 0..2 {
                let physical_index = kx + 6 * ky + 3 * kz;
                storage[physical_index] = shifted_mode(kx, ky, kz, shift);
            }
        }
    }
    Array3::new(layout, VecStorage::new(storage)).expect("strided mode array")
}

#[test]
fn type1_rejects_input_length_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = NufftWgpuPlan1D::new(UniformDomain1D::new(8, 0.25).expect("domain"), 2, 6);
    let error = backend
        .execute_type1_1d(&plan, &[0.0, 0.25], &[Complex32::new(1.0, 0.0)])
        .expect_err("length mismatch must fail");
    assert_input_length_mismatch(error, 2, 1);
}

#[test]
fn fast_type1_reusable_1d_rejects_sample_capacity_overflow() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = NufftWgpuPlan1D::new(UniformDomain1D::new(8, 0.25).expect("domain"), 2, 6);
    let mut buffers =
        NufftGpuBuffers1D::new(backend.device(), &plan, 1).expect("provider buffer allocation");
    let mut output = vec![eunomia::Complex64::new(0.0, 0.0); 8];
    let error = backend
        .execute_fast_type1_1d_with_buffers(
            &mut buffers,
            &[0.0, 0.25],
            &[Complex32::new(1.0, 0.0), Complex32::new(0.5, -0.25)],
            &mut output,
        )
        .expect_err("sample capacity overflow must fail");
    assert_input_length_mismatch(error, 1, 2);
}

#[test]
fn fast_type2_reusable_1d_supports_more_samples_than_modes() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = NufftWgpuPlan1D::new(UniformDomain1D::new(8, 0.25).expect("domain"), 2, 6);
    let coefficients = [
        Complex32::new(1.0, 0.0),
        Complex32::new(0.5, -0.25),
        Complex32::new(-0.75, 0.5),
        Complex32::new(0.25, 0.75),
        Complex32::new(-0.5, -0.1),
        Complex32::new(0.125, 0.25),
        Complex32::new(0.8, -0.6),
        Complex32::new(-0.3, 0.4),
    ];
    let positions = [
        0.0_f32, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6,
    ];
    let expected = backend
        .execute_fast_type2_1d(&plan, &coefficients, &positions)
        .expect("non-reusable fast type2");
    let mut buffers = NufftGpuBuffers1D::new(backend.device(), &plan, positions.len())
        .expect("provider buffer allocation");
    let mut actual = vec![eunomia::Complex64::new(0.0, 0.0); positions.len()];
    backend
        .execute_fast_type2_1d_with_buffers(&mut buffers, &coefficients, &positions, &mut actual)
        .expect("reusable fast type2");
    assert_eq!(actual, expected);
}

#[test]
fn fast_type1_reusable_3d_rejects_sample_capacity_overflow() {
    let Some(backend) = backend() else {
        return;
    };
    let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).expect("grid");
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let mut buffers =
        NufftGpuBuffers3D::new(backend.device(), &plan, 1).expect("provider buffer allocation");
    let mut output = vec![eunomia::Complex64::new(0.0, 0.0); 12];
    let error = backend
        .execute_fast_type1_3d_with_buffers(
            &mut buffers,
            &[(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5)],
            &[Complex32::new(1.0, 0.0), Complex32::new(-0.25, 0.5)],
            &mut output,
        )
        .expect_err("sample capacity overflow must fail");
    assert_input_length_mismatch(error, 1, 2);
}

#[test]
fn fast_type2_reusable_3d_matches_per_call_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let grid = grid3d();
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let modes = modes3d(grid);
    let positions = positions3d();
    let expected = backend
        .execute_fast_type2_3d(&plan, &modes, &positions)
        .expect("per-call fast type2");
    let mut buffers = NufftGpuBuffers3D::new(backend.device(), &plan, positions.len())
        .expect("provider buffer allocation");
    let mut actual = vec![eunomia::Complex64::new(0.0, 0.0); positions.len()];
    backend
        .execute_fast_type2_3d_with_buffers(&mut buffers, &modes, &positions, &mut actual)
        .expect("reusable fast type2");
    assert_eq!(actual, expected);
}

#[test]
fn reusable_3d_strided_dispatch_reuses_host_staging() {
    let Some(backend) = backend() else {
        return;
    };
    let grid = grid3d();
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let positions = positions3d();
    let first_modes = strided_shifted_modes3d(0.0);
    let second_modes = strided_shifted_modes3d(0.375);
    let expected_first = backend
        .execute_fast_type2_3d(&plan, &shifted_modes3d(0.0), &positions)
        .expect("first per-call fast type2");
    let expected_second = backend
        .execute_fast_type2_3d(&plan, &shifted_modes3d(0.375), &positions)
        .expect("second per-call fast type2");
    let mut buffers = NufftGpuBuffers3D::new(backend.device(), &plan, positions.len())
        .expect("provider buffer allocation");
    let mut output = vec![eunomia::Complex64::new(0.0, 0.0); positions.len()];
    backend
        .execute_fast_type2_3d_with_buffers(&mut buffers, &first_modes, &positions, &mut output)
        .expect("warm-up strided dispatch");
    assert_eq!(output, expected_first);

    backend
        .execute_fast_type2_3d_with_buffers(&mut buffers, &second_modes, &positions, &mut output)
        .expect("second strided dispatch");
    assert_eq!(output, expected_second);
    assert_ne!(expected_first, expected_second);
}
