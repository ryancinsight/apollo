//! Unit tests for Discrete Hartley Transform plan.

use super::plan::DhtPlan;
use crate::domain::contracts::error::DhtError;
use apollo_fft::{f16, PrecisionProfile};
use approx::assert_abs_diff_eq;
use leto::{SliceArg, Storage};
use leto::{Array2, Array3};

#[test]
fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
    let plan = DhtPlan::new(8).expect("valid plan");
    let signal64 = [1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let expected = plan.forward(&signal64).expect("forward");
    let expected_inverse = plan.inverse(&expected).expect("inverse");

    let mut out64 = [0.0_f64; 8];
    plan.forward_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
        .expect("typed f64 forward");
    for (actual, expected) in out64.iter().zip(expected.values()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }

    let signal32 = signal64.map(|value| value as f32);
    let mut out32 = [0.0_f32; 8];
    plan.forward_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
        .expect("typed f32 forward");
    for (actual, expected) in out32.iter().zip(expected.values()) {
        assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
    }
    let mut inv32 = [0.0_f32; 8];
    plan.inverse_typed_into(&out32, &mut inv32, PrecisionProfile::LOW_PRECISION_F32)
        .expect("typed f32 inverse");
    for (actual, expected) in inv32.iter().zip(expected_inverse.iter()) {
        assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
    }

    let signal16 = signal64.map(|value| f16::from_f32(value as f32));
    let mut out16 = [f16::from_f32(0.0); 8];
    plan.forward_typed_into(
        &signal16,
        &mut out16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed mixed f16 forward");
    for (actual, expected) in out16.iter().zip(expected.values()) {
        let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
    }
}

#[test]
fn typed_path_rejects_profile_storage_mismatch() {
    let plan = DhtPlan::new(4).expect("valid plan");
    let signal = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    assert!(matches!(
        plan.forward_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
        Err(DhtError::PrecisionMismatch)
    ));
}

#[test]
fn leto_2d_forward_matches_ndarray_reference() {
    let plan = DhtPlan::new(3).expect("valid plan");
    let input = Array2::from_shape_vec(
        [3, 3],
        vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0, -0.75],
    )
    .expect("ndarray input");
    let expected = plan.forward_2d(&input).expect("ndarray forward");

    let leto_input =
        leto::Array2::from_shape_vec([3, 3], input.iter().copied().collect()).expect("leto input");
    let actual = plan
        .forward_2d_leto(leto_input.view())
        .expect("leto forward");

    assert_eq!(actual.shape(), [3, 3]);
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_2d_strided_inverse_matches_ndarray_reference() {
    let plan = DhtPlan::new(3).expect("valid plan");
    let dense = Array2::from_shape_vec(
        [3, 3],
        vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0, -0.75],
    )
    .expect("dense input");
    let spectrum = plan.forward_2d(&dense).expect("ndarray forward");
    let expected = plan.inverse_2d(&spectrum).expect("ndarray inverse");

    let mut interleaved = Vec::with_capacity(18);
    for value in spectrum.iter() {
        interleaved.push(*value);
        interleaved.push(-999.0);
    }
    let leto_input =
        leto::Array2::from_shape_vec([3, 6], interleaved).expect("leto interleaved input");
    let strided = leto_input
        .view()
        .slice_with::<2>(&[
            SliceArg::range(Some(0), Some(3), 1),
            SliceArg::range(Some(0), Some(6), 2),
        ])
        .expect("strided Leto view");
    let actual = plan.inverse_2d_leto(strided).expect("leto inverse");

    assert_eq!(actual.shape(), [3, 3]);
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_3d_forward_matches_ndarray_reference() {
    let plan = DhtPlan::new(2).expect("valid plan");
    let input = Array3::from_shape_vec([2, 2, 2], vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0])
        .expect("ndarray input");
    let expected = plan.forward_3d(&input).expect("ndarray forward");

    let leto_input = leto::Array3::from_shape_vec([2, 2, 2], input.iter().copied().collect())
        .expect("leto input");
    let actual = plan
        .forward_3d_leto(leto_input.view())
        .expect("leto forward");

    assert_eq!(actual.shape(), [2, 2, 2]);
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_3d_inverse_matches_ndarray_reference() {
    let plan = DhtPlan::new(2).expect("valid plan");
    let input = Array3::from_shape_vec([2, 2, 2], vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0])
        .expect("ndarray input");
    let spectrum = plan.forward_3d(&input).expect("ndarray forward");
    let expected = plan.inverse_3d(&spectrum).expect("ndarray inverse");

    let leto_input = leto::Array3::from_shape_vec([2, 2, 2], spectrum.iter().copied().collect())
        .expect("leto input");
    let actual = plan
        .inverse_3d_leto(leto_input.view())
        .expect("leto inverse");

    assert_eq!(actual.shape(), [2, 2, 2]);
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
}
