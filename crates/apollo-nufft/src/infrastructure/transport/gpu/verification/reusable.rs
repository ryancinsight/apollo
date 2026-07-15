//! Reusable-buffer capacity and value-equivalence contracts.

use eunomia::Complex32;

use crate::{
    infrastructure::transport::gpu::{
        NufftGpuBuffers1D, NufftGpuBuffers3D, NufftWgpuPlan1D, NufftWgpuPlan3D,
    },
    UniformDomain1D, UniformGrid3D,
};

use super::support::{assert_input_length_mismatch, backend};

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
    let buffers =
        NufftGpuBuffers1D::new(backend.device(), 8, 16, 1).expect("provider buffer allocation");
    let error = backend
        .execute_fast_type1_1d_with_buffers(
            &plan,
            &buffers,
            &[0.0, 0.25],
            &[Complex32::new(1.0, 0.0), Complex32::new(0.5, -0.25)],
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
    let buffers = NufftGpuBuffers1D::new(backend.device(), 8, 16, positions.len())
        .expect("provider buffer allocation");
    let actual = backend
        .execute_fast_type2_1d_with_buffers(&plan, &buffers, &coefficients, &positions)
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
    let buffers = NufftGpuBuffers3D::new(backend.device(), (3, 2, 2), (16, 16, 16), 1)
        .expect("provider buffer allocation");
    let error = backend
        .execute_fast_type1_3d_with_buffers(
            &plan,
            &buffers,
            &[(0.0_f32, 0.0, 0.0), (0.35, 0.7, 0.5)],
            &[Complex32::new(1.0, 0.0), Complex32::new(-0.25, 0.5)],
        )
        .expect_err("sample capacity overflow must fail");
    assert_input_length_mismatch(error, 1, 2);
}
