//! Typed storage paths and their reusable bridge workspaces.

use super::super::{StftPlan, typed_workspace_capacities};
use crate::domain::contracts::error::StftError;
use apollo_fft::{F16, PrecisionProfile};
use eunomia::{Complex32, Complex64, assert_relative_eq};
use leto::Array1;

#[test]
fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal64 = Array1::from((0..16).map(|i| (i as f64 * 0.2).sin()).collect::<Vec<_>>());
    let expected = plan.forward(&signal64).expect("forward");

    let mut out64 = Array1::<Complex64>::zeros([expected.size()]);
    plan.forward_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
        .expect("typed f64 forward");
    for (actual, expected) in out64.iter().zip(expected.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }

    let signal32 = signal64.mapv(|value| value as f32);
    let represented32 = signal32.mapv(f64::from);
    let expected32 = plan
        .forward(&represented32)
        .expect("represented f32 forward");
    let mut out32 = Array1::<Complex32>::zeros([expected32.size()]);
    plan.forward_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
        .expect("typed f32 forward");
    for (actual, expected) in out32.iter().zip(expected32.iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
        assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
    }

    let mut recovered32 = Array1::<f32>::zeros([signal32.size()]);
    plan.inverse_typed_into(
        &out32,
        signal32.size(),
        &mut recovered32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed f32 inverse");
    for (actual, expected) in recovered32.iter().zip(signal32.iter()) {
        assert!((*actual - *expected).abs() < 1.0e-4);
    }

    let signal16 = signal64.mapv(|value| F16::from_f32(value as f32));
    let represented16 = signal16.mapv(|value| f64::from(value.to_f32()));
    let expected16 = plan
        .forward(&represented16)
        .expect("represented F16 forward");
    let mut out16 = Array1::from_elem([expected16.size()], [F16::from_f32(0.0); 2]);
    plan.forward_typed_into(
        &signal16,
        &mut out16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed F16 forward");
    for (actual, expected) in out16.iter().zip(expected16.iter()) {
        let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
        assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
    }
}

#[test]
fn typed_paths_reuse_bridge_workspace_capacity() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from((0..16).map(|i| (i as f32 * 0.2).sin()).collect::<Vec<_>>());
    let spectrum_len = plan.frame_count(signal.size()) * plan.spectrum_len();
    let mut first_spectrum = Array1::<Complex32>::zeros([spectrum_len]);
    let mut second_spectrum = Array1::<Complex32>::zeros([spectrum_len]);

    plan.forward_typed_into(
        &signal,
        &mut first_spectrum,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("first typed forward");
    let after_first_forward = typed_workspace_capacities();
    assert!(after_first_forward.0 >= signal.size());
    assert!(after_first_forward.2 >= spectrum_len);

    plan.forward_typed_into(
        &signal,
        &mut second_spectrum,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("second typed forward");
    assert_eq!(typed_workspace_capacities(), after_first_forward);
    for (first, second) in first_spectrum.iter().zip(second_spectrum.iter()) {
        assert_eq!(first.re.to_bits(), second.re.to_bits());
        assert_eq!(first.im.to_bits(), second.im.to_bits());
    }

    let mut first_recovered = Array1::<f32>::zeros([signal.size()]);
    let mut second_recovered = Array1::<f32>::zeros([signal.size()]);
    plan.inverse_typed_into(
        &first_spectrum,
        signal.size(),
        &mut first_recovered,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("first typed inverse");
    let after_first_inverse = typed_workspace_capacities();
    assert!(after_first_inverse.1 >= spectrum_len);
    assert!(after_first_inverse.3 >= signal.size());

    plan.inverse_typed_into(
        &first_spectrum,
        signal.size(),
        &mut second_recovered,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("second typed inverse");
    assert_eq!(typed_workspace_capacities(), after_first_inverse);
    for ((first, second), expected) in first_recovered
        .iter()
        .zip(second_recovered.iter())
        .zip(signal.iter())
    {
        assert_eq!(first.to_bits(), second.to_bits());
        assert!((*first - *expected).abs() < 1.0e-4);
    }
}

#[test]
fn typed_path_rejects_profile_storage_mismatch() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from(vec![1.0_f32; 16]);
    let mut output =
        Array1::<Complex32>::zeros([plan.frame_count(signal.size()) * plan.spectrum_len()]);
    assert!(matches!(
        plan.forward_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
        Err(StftError::PrecisionMismatch)
    ));
}
