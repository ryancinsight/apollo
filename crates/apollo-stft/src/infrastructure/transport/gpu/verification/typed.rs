//! Value-semantic STFT GPU verification for one bounded contract.

use crate::infrastructure::transport::gpu::{StftWgpuPlan, WgpuError};
use apollo_fft::{f16, PrecisionProfile};
use leto::{SliceArg, Storage};

use super::support::backend;

#[test]
fn stft_wgpu_typed_mixed_storage() {
    let Some(backend) = backend() else {
        return;
    };
    // 8-sample signal: f16 quantization of the alternating pattern is exact for these values.
    let signal_f32: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
    let signal_f16: Vec<f16> = signal_f32.iter().map(|&x| f16::from_f32(x)).collect();
    // Represented input: f16 promoted back to f32 (round-trip defines the reference).
    let represented: Vec<f32> = signal_f16.iter().map(|v| v.to_f32()).collect();
    let plan = StftWgpuPlan::new(4, 2);
    // frame_count = 1 + 8.div_ceil(2) = 5; output_len = 5 * 4 = 20.
    let f32_result = backend
        .execute_forward(&plan, &represented)
        .expect("f32 reference");
    let mut typed_out: Vec<[f16; 2]> =
        vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; f32_result.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &signal_f16,
            &mut typed_out,
        )
        .expect("typed mixed forward");
    for (actual, expected) in typed_out.iter().zip(f32_result.iter()) {
        let expected_f16 = [f16::from_f32(expected.re), f16::from_f32(expected.im)];
        assert_eq!(
            actual[0].to_bits(),
            expected_f16[0].to_bits(),
            "re bits mismatch: actual={:?} expected={:?}",
            actual[0],
            expected_f16[0]
        );
        assert_eq!(
            actual[1].to_bits(),
            expected_f16[1].to_bits(),
            "im bits mismatch: actual={:?} expected={:?}",
            actual[1],
            expected_f16[1]
        );
    }
}

#[test]
fn stft_wgpu_leto_forward_and_inverse() {
    let Some(backend) = backend() else {
        return;
    };
    let signal: Vec<f32> = vec![
        0.0, 1.0, 0.0, -1.0, 0.25, 0.75, -0.25, -0.75, 0.0, 1.0, 0.0, -1.0, 0.25, 0.75, -0.25,
        -0.75,
    ];
    let plan = StftWgpuPlan::new(8, 4);
    let leto_signal = leto::Array1::from_shape_vec([signal.len()], signal.clone()).expect("signal");

    let expected_forward = backend
        .execute_forward(&plan, &signal)
        .expect("slice forward");
    let actual_forward = backend
        .execute_forward_leto(&plan, leto_signal.view())
        .expect("leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let leto_spectrum =
        leto::Array1::from_shape_vec([expected_forward.len()], expected_forward.clone())
            .expect("spectrum");
    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward, signal.len())
        .expect("slice inverse");
    let actual_inverse = backend
        .execute_inverse_leto(&plan, leto_spectrum.view(), signal.len())
        .expect("leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn stft_wgpu_leto_strided_forward() {
    let Some(backend) = backend() else {
        return;
    };
    let logical: Vec<f32> = vec![
        0.0, 1.0, 0.0, -1.0, 0.25, 0.75, -0.25, -0.75, 0.0, 1.0, 0.0, -1.0, 0.25, 0.75, -0.25,
        -0.75,
    ];
    let mut backing = Vec::with_capacity(logical.len() * 2);
    for value in &logical {
        backing.push(*value);
        backing.push(99.0);
    }
    let leto_signal = leto::Array1::from_shape_vec([backing.len()], backing).expect("signal");
    let strided = leto_signal
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided signal");

    let plan = StftWgpuPlan::new(8, 4);
    let expected = backend
        .execute_forward(&plan, &logical)
        .expect("slice forward");
    let actual = backend
        .execute_forward_leto(&plan, strided)
        .expect("leto strided forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}

#[test]
fn stft_wgpu_typed_leto() {
    let Some(backend) = backend() else {
        return;
    };
    let signal_f16: Vec<f16> = [0.0_f32, 1.0, 0.0, -1.0, 0.25, 0.75, -0.25, -0.75]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let plan = StftWgpuPlan::new(4, 2);
    let frame_count = 1 + signal_f16.len().div_ceil(plan.hop_len());
    let output_len = frame_count * plan.frame_len();

    let mut expected_forward: Vec<[f16; 2]> =
        vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; output_len];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &signal_f16,
            &mut expected_forward,
        )
        .expect("typed slice forward");

    let leto_signal =
        leto::Array1::from_shape_vec([signal_f16.len()], signal_f16.clone()).expect("signal");
    let actual_forward = backend
        .execute_forward_leto_typed::<f16, [f16; 2]>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_signal.view(),
        )
        .expect("typed leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let mut expected_inverse = vec![f16::from_f32(0.0); signal_f16.len()];
    backend
        .execute_inverse_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &expected_forward,
            signal_f16.len(),
            &mut expected_inverse,
        )
        .expect("typed slice inverse");

    let actual_inverse = backend
        .execute_inverse_leto_typed::<[f16; 2], f16>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            actual_forward.view(),
            signal_f16.len(),
        )
        .expect("typed leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn stft_wgpu_typed_profile_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    // f16 signal: I::PROFILE == MIXED_PRECISION_F16_F32.
    // Passing LOW_PRECISION_F32 as input_precision must trigger InvalidPrecisionProfile
    // before any GPU work is attempted.
    let signal_f16: Vec<f16> = vec![
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(-1.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(-1.0),
    ];
    let plan = StftWgpuPlan::new(4, 2);
    let frame_count = 1 + signal_f16.len().div_ceil(plan.hop_len());
    let output_len = frame_count * plan.frame_len();
    let mut out: Vec<[f16; 2]> = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; output_len];
    let error = backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &signal_f16,
            &mut out,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(error, WgpuError::InvalidPrecisionProfile));
}
