//! Value-semantic Wavelet GPU represented-storage contracts.

use leto::Storage;

use crate::infrastructure::transport::gpu::{WaveletWgpuPlan, WgpuError};

use super::support::backend;

#[test]
fn typed_mixed_storage_matches_represented_f32_execution_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    use apollo_fft::{f16, PrecisionProfile};
    let represented = [1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
    let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
    let represented_input: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
    let plan = WaveletWgpuPlan::new(input.len(), 3);
    let expected_fwd = backend
        .execute_forward(&plan, &represented_input)
        .expect("represented forward");
    let mut typed_fwd = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut typed_fwd,
        )
        .expect("typed mixed forward");
    assert_eq!(typed_fwd.len(), expected_fwd.len());
    for (actual, expected) in typed_fwd.iter().zip(expected_fwd.iter()) {
        let expected_f16 = f16::from_f32(*expected);
        assert_eq!(actual.to_bits(), expected_f16.to_bits());
    }

    let expected_inv = backend
        .execute_inverse(&plan, &expected_fwd)
        .expect("represented inverse");
    let mut typed_inv = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_inverse_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &typed_fwd,
            &mut typed_inv,
        )
        .expect("typed mixed inverse");
    for (actual, expected) in typed_inv.iter().zip(expected_inv.iter()) {
        let q = expected.abs() * 2.0_f32.powi(-10) + f32::from(f16::MIN_POSITIVE);
        assert!(
            (actual.to_f32() - expected).abs() <= q,
            "f16 quantization mismatch: actual={}, expected={}",
            actual.to_f32(),
            expected
        );
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_typed_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    use apollo_fft::{f16, PrecisionProfile};
    let represented = [1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
    let input: Vec<f16> = represented.iter().copied().map(f16::from_f32).collect();
    let plan = WaveletWgpuPlan::new(input.len(), 3);

    let mut expected_forward = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut expected_forward,
        )
        .expect("typed slice forward");
    let input_leto = leto::Array1::from_shape_vec([input.len()], input).expect("input");
    let actual_forward = backend
        .execute_forward_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            input_leto.view(),
        )
        .expect("typed leto forward");
    assert_eq!(
        actual_forward
            .storage()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_forward
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let mut expected_inverse = vec![f16::from_f32(0.0); expected_forward.len()];
    backend
        .execute_inverse_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &expected_forward,
            &mut expected_inverse,
        )
        .expect("typed slice inverse");
    let coeffs_leto = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("coefficients");
    let actual_inverse = backend
        .execute_inverse_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            coeffs_leto.view(),
        )
        .expect("typed leto inverse");
    assert_eq!(
        actual_inverse
            .storage()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_inverse
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn typed_path_rejects_profile_mismatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    use apollo_fft::{f16, PrecisionProfile};
    let plan = WaveletWgpuPlan::new(8, 3);
    let input = vec![f16::from_f32(1.0); 8];
    let mut output = vec![f16::from_f32(0.0); 8];
    let err = backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::LOW_PRECISION_F32,
            &input,
            &mut output,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(err, WgpuError::InvalidPrecisionProfile));
}
