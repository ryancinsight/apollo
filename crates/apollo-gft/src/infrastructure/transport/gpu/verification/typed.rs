//! Value-semantic GFT GPU represented-storage contracts.

use apollo_fft::{f16, PrecisionProfile};
use leto::Storage;

use super::support::{backend, path4_plan_and_basis};

#[test]
fn typed_leto_forward_and_inverse_match_typed_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (_cpu_plan, basis, signal) = path4_plan_and_basis();
    let plan = crate::infrastructure::transport::gpu::GftWgpuPlan::new(4);
    let input: Vec<f16> = signal.iter().copied().map(f16::from_f32).collect();
    let basis_leto = leto::Array1::from_shape_vec([basis.len()], basis).expect("basis");
    let mut expected_forward = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            basis_leto.storage().as_slice(),
            &mut expected_forward,
        )
        .expect("typed slice forward");
    let input = leto::Array1::from_shape_vec([input.len()], input).expect("input");
    let actual_forward = backend
        .execute_forward_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            input.view(),
            basis_leto.view(),
        )
        .expect("typed Leto forward");
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
            basis_leto.storage().as_slice(),
            &mut expected_inverse,
        )
        .expect("typed slice inverse");
    let spectrum =
        leto::Array1::from_shape_vec([expected_forward.len()], expected_forward).expect("spectrum");
    let actual_inverse = backend
        .execute_inverse_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            spectrum.view(),
            basis_leto.view(),
        )
        .expect("typed Leto inverse");
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
fn typed_mixed_storage_matches_represented_execution_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (_cpu_plan, basis, signal) = path4_plan_and_basis();
    let input: Vec<f16> = signal.iter().copied().map(f16::from_f32).collect();
    let represented: Vec<f32> = input.iter().map(|value| value.to_f32()).collect();
    let plan = crate::infrastructure::transport::gpu::GftWgpuPlan::new(4);
    let expected_forward = backend
        .execute_forward(&plan, &represented, &basis)
        .expect("represented forward");
    let mut actual_forward = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_forward_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &input,
            &basis,
            &mut actual_forward,
        )
        .expect("typed mixed forward");
    assert_eq!(actual_forward.len(), expected_forward.len());
    for (actual, expected) in actual_forward.iter().zip(expected_forward.iter()) {
        assert_eq!(actual.to_bits(), f16::from_f32(*expected).to_bits());
    }

    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward, &basis)
        .expect("represented inverse");
    let mut actual_inverse = vec![f16::from_f32(0.0); input.len()];
    backend
        .execute_inverse_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &actual_forward,
            &basis,
            &mut actual_inverse,
        )
        .expect("typed mixed inverse");
    for (actual, expected) in actual_inverse.iter().zip(expected_inverse.iter()) {
        let quantization_bound = expected.abs() * 2.0_f32.powi(-10) + f32::from(f16::MIN_POSITIVE);
        assert!(
            (actual.to_f32() - expected).abs() <= quantization_bound,
            "f16 quantization mismatch: actual={}, expected={expected}",
            actual.to_f32()
        );
    }
}
