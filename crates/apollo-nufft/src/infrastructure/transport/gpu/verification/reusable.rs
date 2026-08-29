//! Reusable-buffer capacity and value-equivalence contracts.

use eunomia::{Complex32, Complex64};
use hephaestus_core::HephaestusError;
use leto::{Array3, Layout, Storage, VecStorage};

use crate::{
    infrastructure::transport::gpu::{
        NufftGpuBuffers1D, NufftGpuBuffers3D, NufftWgpuError, NufftWgpuPlan1D, NufftWgpuPlan3D,
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

fn assert_foreign_device_error(error: NufftWgpuError) {
    match error {
        NufftWgpuError::Provider(HephaestusError::DispatchFailed { message }) => {
            assert_eq!(message, "prepared WGPU FFT belongs to a different device")
        }
        other => panic!("expected foreign-device provider error, received {other:?}"),
    }
}

#[test]
fn reusable_workspaces_reject_foreign_devices_before_output_mutation() {
    let Some(owner) = backend() else {
        return;
    };
    let foreign = backend().expect("an available adapter must create a second logical device");
    let plan_1d = NufftWgpuPlan1D::new(UniformDomain1D::new(8, 0.25).expect("domain"), 2, 6);
    let positions_1d = [0.0_f32, 0.25];
    let values_1d = [Complex32::new(1.0, 0.0), Complex32::new(-0.25, 0.5)];
    let coefficients_1d = [Complex32::new(0.5, -0.25); 8];
    let mut buffers_1d = NufftGpuBuffers1D::new(owner.device(), &plan_1d, positions_1d.len())
        .expect("owner buffer allocation");
    let sentinel = Complex64::new(17.0, -23.0);
    let mut type1_output_1d = vec![sentinel; 8];
    let error = foreign
        .execute_fast_type1_1d_with_buffers(
            &mut buffers_1d,
            &positions_1d,
            &values_1d,
            &mut type1_output_1d,
        )
        .expect_err("foreign Type-1 1D workspace must fail");
    assert_foreign_device_error(error);
    assert_eq!(type1_output_1d, vec![sentinel; 8]);

    let mut type2_output_1d = vec![sentinel; positions_1d.len()];
    let error = foreign
        .execute_fast_type2_1d_with_buffers(
            &mut buffers_1d,
            &coefficients_1d,
            &positions_1d,
            &mut type2_output_1d,
        )
        .expect_err("foreign Type-2 1D workspace must fail");
    assert_foreign_device_error(error);
    assert_eq!(type2_output_1d, vec![sentinel; positions_1d.len()]);

    let grid = grid3d();
    let plan_3d = NufftWgpuPlan3D::new(grid, 2, 6);
    let positions_3d = positions3d();
    let values_3d = [Complex32::new(1.0, 0.0); 3];
    let modes_3d = modes3d(grid);
    let mut buffers_3d = NufftGpuBuffers3D::new(owner.device(), &plan_3d, positions_3d.len())
        .expect("owner buffer allocation");
    let mut type1_output_3d = vec![sentinel; 12];
    let error = foreign
        .execute_fast_type1_3d_with_buffers(
            &mut buffers_3d,
            &positions_3d,
            &values_3d,
            &mut type1_output_3d,
        )
        .expect_err("foreign Type-1 3D workspace must fail");
    assert_foreign_device_error(error);
    assert_eq!(type1_output_3d, vec![sentinel; 12]);

    let mut type2_output_3d = vec![sentinel; positions_3d.len()];
    let error = foreign
        .execute_fast_type2_3d_with_buffers(
            &mut buffers_3d,
            &modes_3d,
            &positions_3d,
            &mut type2_output_3d,
        )
        .expect_err("foreign Type-2 3D workspace must fail");
    assert_foreign_device_error(error);
    assert_eq!(type2_output_3d, vec![sentinel; positions_3d.len()]);
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
fn fast_type1_reusable_1d_updates_logical_sample_count() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = NufftWgpuPlan1D::new(UniformDomain1D::new(8, 0.25).expect("domain"), 2, 6);
    let first_positions = [0.0_f32, 0.25, 0.5, 0.75];
    let first_values = [
        Complex32::new(1.0, 0.0),
        Complex32::new(-0.25, 0.5),
        Complex32::new(0.75, -0.5),
        Complex32::new(0.125, 0.25),
    ];
    let second_positions = [0.1_f32, 0.6];
    let second_values = [Complex32::new(-0.5, 0.25), Complex32::new(0.75, 0.125)];
    let expected_first = backend
        .execute_fast_type1_1d(&plan, &first_positions, &first_values)
        .expect("first per-call fast type1");
    let expected_second = backend
        .execute_fast_type1_1d(&plan, &second_positions, &second_values)
        .expect("second per-call fast type1");
    let mut buffers = NufftGpuBuffers1D::new(backend.device(), &plan, first_positions.len())
        .expect("provider buffer allocation");
    let mut actual = vec![eunomia::Complex64::new(0.0, 0.0); expected_first.len()];

    backend
        .execute_fast_type1_1d_with_buffers(
            &mut buffers,
            &first_positions,
            &first_values,
            &mut actual,
        )
        .expect("first reusable fast type1");
    assert_eq!(actual, expected_first.storage().as_slice());

    backend
        .execute_fast_type1_1d_with_buffers(
            &mut buffers,
            &second_positions,
            &second_values,
            &mut actual,
        )
        .expect("second reusable fast type1");
    assert_eq!(actual, expected_second.storage().as_slice());
    assert_ne!(expected_first, expected_second);
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

    let shorter_positions = &positions[..5];
    let expected_shorter = backend
        .execute_fast_type2_1d(&plan, &coefficients, shorter_positions)
        .expect("shorter per-call fast type2");
    let mut actual_shorter = vec![eunomia::Complex64::new(0.0, 0.0); shorter_positions.len()];
    backend
        .execute_fast_type2_1d_with_buffers(
            &mut buffers,
            &coefficients,
            shorter_positions,
            &mut actual_shorter,
        )
        .expect("shorter reusable fast type2");
    assert_eq!(actual_shorter, expected_shorter);
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
fn fast_type1_reusable_3d_updates_logical_sample_count() {
    let Some(backend) = backend() else {
        return;
    };
    let grid = grid3d();
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    let first_positions = positions3d();
    let first_values = [
        Complex32::new(1.0, 0.0),
        Complex32::new(-0.25, 0.5),
        Complex32::new(0.75, -0.5),
    ];
    let second_positions = [(0.15_f32, 0.3, 0.45)];
    let second_values = [Complex32::new(-0.5, 0.25)];
    let expected_first = backend
        .execute_fast_type1_3d(&plan, &first_positions, &first_values)
        .expect("first per-call fast type1");
    let expected_second = backend
        .execute_fast_type1_3d(&plan, &second_positions, &second_values)
        .expect("second per-call fast type1");
    let mut buffers = NufftGpuBuffers3D::new(backend.device(), &plan, first_positions.len())
        .expect("provider buffer allocation");
    let mut actual = vec![eunomia::Complex64::new(0.0, 0.0); expected_first.len()];

    backend
        .execute_fast_type1_3d_with_buffers(
            &mut buffers,
            &first_positions,
            &first_values,
            &mut actual,
        )
        .expect("first reusable fast type1");
    assert_eq!(actual, expected_first.storage().as_slice());

    backend
        .execute_fast_type1_3d_with_buffers(
            &mut buffers,
            &second_positions,
            &second_values,
            &mut actual,
        )
        .expect("second reusable fast type1");
    assert_eq!(actual, expected_second.storage().as_slice());
    assert_ne!(expected_first, expected_second);
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

    let shorter_positions = &positions[..2];
    let expected_shorter = backend
        .execute_fast_type2_3d(&plan, &modes, shorter_positions)
        .expect("shorter per-call fast type2");
    let mut actual_shorter = vec![eunomia::Complex64::new(0.0, 0.0); shorter_positions.len()];
    backend
        .execute_fast_type2_3d_with_buffers(
            &mut buffers,
            &modes,
            shorter_positions,
            &mut actual_shorter,
        )
        .expect("shorter reusable fast type2");
    assert_eq!(actual_shorter, expected_shorter);
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
