//! Value-semantic Hilbert GPU represented-storage contracts.

use apollo_fft::{f16, PrecisionProfile};
use leto::Storage;

use super::support::backend;

#[test]
fn typed_mixed_storage_matches_represented_execution_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let represented = [1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
    let represented_input: Vec<f32> = input.iter().map(|value| value.to_f64() as f32).collect();
    let plan = backend.plan(input.len());
    let expected = backend
        .execute_forward(&plan, &represented_input)
        .expect("represented forward");
    let mut actual = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut actual,
        )
        .expect("typed mixed forward");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_eq!(actual.to_bits(), f16::from_f32(*expected).to_bits());
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_typed_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let represented = [1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
    let plan = backend.plan(input.len());
    let mut expected_forward = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut expected_forward,
        )
        .expect("typed forward");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto typed input");
    let actual_forward = backend
        .execute_forward_leto_typed::<f16>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_input.view(),
        )
        .expect("Leto typed forward");
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
    let leto_quadrature = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("Leto typed quadrature");
    let actual_inverse = backend
        .execute_inverse_leto_typed::<f16>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_quadrature.view(),
        )
        .expect("Leto typed inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}
