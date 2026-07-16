use apollo_fft::{f16, PrecisionProfile};
use eunomia::Complex32;
use leto::{Array1, Storage};

use super::support::backend;

fn mixed_input() -> Vec<[f16; 2]> {
    vec![
        [f16::from_f32(1.0), f16::from_f32(0.0)],
        [f16::from_f32(-0.5), f16::from_f32(0.75)],
        [f16::from_f32(0.25), f16::from_f32(-1.25)],
        [f16::from_f32(2.0), f16::from_f32(0.5)],
    ]
}

#[test]
fn typed_mixed_storage_matches_represented_f32_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let input = mixed_input();
    let represented_f32: Vec<Complex32> = input
        .iter()
        .map(|[re, im]| Complex32::new(re.to_f32(), im.to_f32()))
        .collect();
    let plan = backend.plan(input.len());
    let expected = backend
        .execute_forward(&plan, &represented_f32)
        .expect("f32 forward");
    let mut typed_output = vec![[f16::from_f32(0.0); 2]; input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut typed_output,
        )
        .expect("typed mixed forward");
    for (actual, expected_value) in typed_output.iter().zip(expected.iter()) {
        assert_eq!(
            actual[0].to_bits(),
            f16::from_f32(expected_value.re).to_bits()
        );
        assert_eq!(
            actual[1].to_bits(),
            f16::from_f32(expected_value.im).to_bits()
        );
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_typed_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let input = mixed_input();
    let leto_input = Array1::from_shape_vec([input.len()], input.clone()).expect("input");
    let plan = backend.plan(input.len());

    let mut expected_forward = vec![[f16::from_f32(0.0); 2]; input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut expected_forward,
        )
        .expect("typed slice forward");
    let actual_forward = backend
        .execute_forward_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_input.view(),
        )
        .expect("typed leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let leto_spectrum = Array1::from_shape_vec([expected_forward.len()], expected_forward.clone())
        .expect("spectrum");
    let mut expected_inverse = vec![[f16::from_f32(0.0); 2]; input.len()];
    backend
        .execute_inverse_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &expected_forward,
            &mut expected_inverse,
        )
        .expect("typed slice inverse");
    let actual_inverse = backend
        .execute_inverse_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_spectrum.view(),
        )
        .expect("typed leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}
