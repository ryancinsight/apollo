//! Value-semantic CZT GPU represented-storage contracts.

use apollo_fft::{f16, PrecisionProfile};
use eunomia::Complex32;
use leto::Storage;

use super::support::{backend, reference_input, reference_parameters};

#[test]
fn typed_mixed_storage_matches_represented_execution_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (a, w) = reference_parameters();
    let input: Vec<[f16; 2]> = reference_input()
        .iter()
        .map(|value| [f16::from_f32(value.re), f16::from_f32(value.im)])
        .collect();
    let represented: Vec<Complex32> = input
        .iter()
        .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
        .collect();
    let plan = backend.plan(input.len(), 6, a, w);
    let expected = backend
        .execute_forward(&plan, &represented)
        .expect("represented forward");
    let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; 6];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut actual,
        )
        .expect("typed mixed forward");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        let expected = [f16::from_f32(expected.re), f16::from_f32(expected.im)];
        assert_eq!(actual[0].to_bits(), expected[0].to_bits());
        assert_eq!(actual[1].to_bits(), expected[1].to_bits());
    }
}

#[test]
fn typed_leto_forward_matches_typed_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (a, w) = reference_parameters();
    let input: Vec<[f16; 2]> = reference_input()
        .iter()
        .map(|value| [f16::from_f32(value.re), f16::from_f32(value.im)])
        .collect();
    let plan = backend.plan(input.len(), 6, a, w);
    let mut expected = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; 6];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut expected,
        )
        .expect("typed slice forward");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto input");
    let actual = backend
        .execute_forward_leto_typed::<[f16; 2]>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_input.view(),
        )
        .expect("typed Leto forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}
