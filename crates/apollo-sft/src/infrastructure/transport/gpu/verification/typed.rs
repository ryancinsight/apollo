//! Value-semantic SFT GPU represented-storage contracts.

use apollo_fft::{f16, PrecisionProfile};
use leto::Storage;

use crate::infrastructure::transport::gpu::SftWgpuPlan;

use super::support::{assert_reference_complex_close, backend, two_tone_signal};

#[test]
fn typed_mixed_storage_forward_matches_represented_execution_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(4, 2);
    let source_signal = two_tone_signal(4, &[(1, 3.0), (2, 1.25)]);
    let native_input: Vec<eunomia::Complex32> = source_signal
        .iter()
        .map(|value| eunomia::Complex32::new(value.re as f32, value.im as f32))
        .collect();
    let mixed_input: Vec<[f16; 2]> = native_input
        .iter()
        .map(|value| [f16::from_f32(value.re), f16::from_f32(value.im)])
        .collect();
    let represented_input: Vec<eunomia::Complex32> = mixed_input
        .iter()
        .map(|value| eunomia::Complex32::new(value[0].to_f32(), value[1].to_f32()))
        .collect();

    let expected = backend
        .execute_forward(&plan, &represented_input)
        .expect("represented f32 forward");
    let actual = backend
        .execute_forward_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &mixed_input,
        )
        .expect("typed mixed forward");

    assert_eq!(actual.frequencies, expected.frequencies);
    assert_eq!(actual.values.len(), expected.values.len());
    for (actual, expected) in actual.values.iter().zip(expected.values.iter()) {
        assert_reference_complex_close(*actual, *expected, 1.0e-3);
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_typed_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(4, 2);
    let source_signal = two_tone_signal(4, &[(1, 3.0), (2, 1.25)]);
    let native_input: Vec<eunomia::Complex32> = source_signal
        .iter()
        .map(|value| eunomia::Complex32::new(value.re as f32, value.im as f32))
        .collect();
    let mixed_input: Vec<[f16; 2]> = native_input
        .iter()
        .map(|value| [f16::from_f32(value.re), f16::from_f32(value.im)])
        .collect();
    let leto_input =
        leto::Array1::from_shape_vec([mixed_input.len()], mixed_input.clone()).expect("input");

    let expected_forward = backend
        .execute_forward_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &mixed_input,
        )
        .expect("typed slice forward");
    let actual_forward = backend
        .execute_forward_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_input.view(),
        )
        .expect("typed Leto forward");
    assert_eq!(actual_forward.frequencies, expected_forward.frequencies);
    assert_eq!(actual_forward.values, expected_forward.values);

    let mut expected_inverse = vec![[f16::from_f32(0.0); 2]; plan.len()];
    backend
        .execute_inverse_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &expected_forward,
            &mut expected_inverse,
        )
        .expect("typed slice inverse");
    let actual_inverse = backend
        .execute_inverse_leto_typed::<[f16; 2]>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &expected_forward,
        )
        .expect("typed Leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}
