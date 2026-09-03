use apollo_fft::{PrecisionProfile, F16};
use eunomia::Complex32;
use leto::Storage;

use crate::infrastructure::transport::gpu::{FrftWgpuPlan, WgpuError};

use super::support::backend;
use crate::infrastructure::transport::gpu::OrderPlan;

fn mixed_input() -> Vec<[F16; 2]> {
    let source_re = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.5, 0.0, -0.75];
    let source_im = [-0.25_f32, 0.75, -1.5, 0.5, 1.0, -0.25, 0.125, -1.0];
    source_re
        .iter()
        .zip(source_im.iter())
        .map(|(&re, &im)| [F16::from_f32(re), F16::from_f32(im)])
        .collect()
}

#[test]
fn mixed_storage_matches_represented_f32_output() {
    let Some(backend) = backend() else {
        return;
    };
    let input = mixed_input();
    let represented = input
        .iter()
        .map(|[re, im]| Complex32::new(re.to_f32(), im.to_f32()))
        .collect::<Vec<_>>();
    let plan = FrftWgpuPlan::new(OrderPlan::new(input.len(), 0.5_f32));
    let expected = backend
        .execute_forward(&plan, &represented)
        .expect("f32 forward reference");
    let mut actual = vec![[F16::from_f32(0.0); 2]; input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &mut actual,
        )
        .expect("typed mixed forward");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        let expected_re = F16::from_f32(expected.re);
        let expected_im = F16::from_f32(expected.im);
        assert_eq!(actual[0].to_bits(), expected_re.to_bits());
        assert_eq!(actual[1].to_bits(), expected_im.to_bits());
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_typed_slice_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let input = mixed_input();
    let plan = FrftWgpuPlan::new(OrderPlan::new(input.len(), 0.5_f32));
    let mut expected_forward = vec![[F16::from_f32(0.0); 2]; input.len()];
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
        .execute_forward_leto_typed::<[F16; 2], [F16; 2]>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_input.view(),
        )
        .expect("Leto typed forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let mut expected_inverse = vec![[F16::from_f32(0.0); 2]; expected_forward.len()];
    backend
        .execute_inverse_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &expected_forward,
            &mut expected_inverse,
        )
        .expect("typed inverse");
    let leto_spectrum = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("Leto typed spectrum");
    let actual_inverse = backend
        .execute_inverse_leto_typed::<[F16; 2], [F16; 2]>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_spectrum.view(),
        )
        .expect("Leto typed inverse");
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
    let plan = FrftWgpuPlan::new(OrderPlan::new(2, 0.5_f32));
    let input = [
        [F16::from_f32(1.0), F16::from_f32(0.0)],
        [F16::from_f32(-1.0), F16::from_f32(0.5)],
    ];
    let mut output = [[F16::from_f32(0.0); 2]; 2];
    let error = backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::LOW_PRECISION_F32,
            &input,
            &mut output,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(error, WgpuError::InvalidPrecisionProfile));
}
