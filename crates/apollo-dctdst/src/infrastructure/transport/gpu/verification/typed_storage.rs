use apollo_fft::{f16, PrecisionProfile};
use leto::Storage;

use crate::{infrastructure::transport::gpu::WgpuError, RealTransformKind};

use super::support::backend;

#[test]
fn mixed_storage_matches_represented_f32_output() {
    let Some(backend) = backend() else {
        return;
    };
    let represented = [0.75_f32, -1.25, 2.0, -0.5, 3.0, 1.5, 0.25, -0.875];
    let input = represented
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
    let represented_input = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
    let plan = backend.plan(input.len(), RealTransformKind::DctII);
    let expected = backend
        .execute_forward(&plan, &represented_input)
        .expect("represented f32 forward");
    let mut actual = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut actual,
        )
        .expect("mixed forward");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_eq!(actual.to_bits(), f16::from_f32(*expected).to_bits());
    }
}

#[test]
fn typed_leto_boundaries_match_typed_slice_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let represented = [0.75_f32, -1.25, 2.0, -0.5, 3.0, 1.5];
    let input = represented
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
    let plan = backend.plan(input.len(), RealTransformKind::DctII);
    let mut expected_forward = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut expected_forward,
        )
        .expect("typed forward");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto input");
    let actual_forward = backend
        .execute_forward_leto_typed::<f16>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_input.view(),
        )
        .expect("typed Leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let mut expected_inverse = vec![f16::from_f32(0.0); expected_forward.len()];
    backend
        .execute_inverse_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &expected_forward,
            &mut expected_inverse,
        )
        .expect("typed inverse");
    let leto_spectrum = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("Leto spectrum");
    let actual_inverse = backend
        .execute_inverse_leto_typed::<f16>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_spectrum.view(),
        )
        .expect("typed Leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn typed_execution_rejects_profile_storage_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = backend.plan(4, RealTransformKind::DctII);
    let input = [1.0_f32, -1.0, 0.5, -0.5];
    let mut output = [0.0_f32; 4];
    let error = backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::HIGH_ACCURACY_F64,
            &input,
            &mut output,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(error, WgpuError::InvalidPrecisionProfile));
}
