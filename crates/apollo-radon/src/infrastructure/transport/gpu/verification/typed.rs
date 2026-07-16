//! Value-semantic Radon GPU represented-storage contracts.

use apollo_fft::{f16, PrecisionProfile};
use leto::{Array2, Storage};

use crate::infrastructure::transport::gpu::WgpuError;

use super::support::backend;

#[test]
fn typed_flat_mixed_storage_matches_represented_f32_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
    let plan = backend.plan(3, 3, angles.len(), 5, 1.0);

    let flat_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let flat_f16: Vec<f16> = flat_f32.iter().copied().map(f16::from_f32).collect();
    let represented_f32: Vec<f32> = flat_f16.iter().map(|v| v.to_f32()).collect();
    let image_represented = Array2::from_shape_vec([3, 3], represented_f32).expect("reshape");

    let expected = backend
        .execute_forward(&plan, &image_represented, &angles)
        .expect("represented f32 forward");
    let actual = backend
        .execute_forward_flat_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &flat_f16,
            &angles,
        )
        .expect("typed flat mixed forward");

    assert_eq!(actual.shape(), expected.shape());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (f64::from(*a) - f64::from(*e)).abs() < 0.1,
            "mismatch at index {index}: actual={a}, expected={e}"
        );
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_typed_flat() {
    let Some(backend) = backend() else {
        return;
    };
    let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
    let angles_leto =
        leto::Array::from_mnemosyne_slice([angles.len()], &angles).expect("leto angles");
    let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
    let flat_f16: Vec<f16> = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let image_leto =
        leto::Array::from_mnemosyne_slice([3, 3], &flat_f16).expect("typed leto image");

    let expected_forward = backend
        .execute_forward_flat_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &flat_f16,
            &angles,
        )
        .expect("typed flat forward");
    let actual_forward = backend
        .execute_forward_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            image_leto.view(),
            angles_leto.view(),
        )
        .expect("typed leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice().expect("contiguous forward")
    );

    let sinogram_f16: Vec<f16> = expected_forward
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect();
    let sinogram_leto = leto::Array::from_mnemosyne_slice(
        [plan.angle_count(), plan.detector_count()],
        &sinogram_f16,
    )
    .expect("typed leto sinogram");
    let expected_inverse = backend
        .execute_inverse_flat_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &sinogram_f16,
            &angles,
        )
        .expect("typed flat inverse");
    let actual_inverse = backend
        .execute_inverse_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            sinogram_leto.view(),
            angles_leto.view(),
        )
        .expect("typed leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice().expect("contiguous inverse")
    );
}

#[test]
fn typed_flat_path_rejects_profile_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
    let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
    let flat_f16: Vec<f16> = vec![f16::from_f32(1.0); plan.rows() * plan.cols()];

    let err = backend
        .execute_forward_flat_typed::<f16>(
            &plan,
            PrecisionProfile::LOW_PRECISION_F32,
            &flat_f16,
            &angles,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(err, WgpuError::InvalidPrecisionProfile));
}
