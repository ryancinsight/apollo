use super::DwtPlan;
use crate::domain::contracts::error::WaveletError;
use crate::domain::metadata::wavelet::DiscreteWavelet;
use apollo_fft::{f16, PrecisionProfile};
use eunomia::assert_abs_diff_eq;

fn detail_buffers<T: Copy>(plan: &DwtPlan, fill: T) -> Vec<Vec<T>> {
    plan.coefficient_shapes()
        .map(|len| vec![fill; len])
        .collect()
}

#[test]
fn typed_dwt_paths_support_f64_f32_and_mixed_f16_storage() {
    let plan = DwtPlan::new(8, 3, DiscreteWavelet::Haar).expect("valid DWT plan");
    let signal64 = [1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let expected = plan.forward(&signal64).expect("forward");

    let mut approx64 = vec![0.0_f64; 1];
    let mut details64 = detail_buffers(&plan, 0.0_f64);
    plan.forward_typed_into(
        &signal64,
        &mut approx64,
        &mut details64,
        PrecisionProfile::HIGH_ACCURACY_F64,
    )
    .expect("typed f64 forward");
    for (actual, expected) in approx64.iter().zip(expected.approximation()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
    for (actual_detail, expected_detail) in details64.iter().zip(expected.details()) {
        for (actual, expected) in actual_detail.iter().zip(expected_detail) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    let signal32 = signal64.map(|value| value as f32);
    let mut approx32 = vec![0.0_f32; 1];
    let mut details32 = detail_buffers(&plan, 0.0_f32);
    plan.forward_typed_into(
        &signal32,
        &mut approx32,
        &mut details32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed f32 forward");
    let expected32 = plan
        .forward(&signal32.map(f64::from))
        .expect("represented f32 forward");
    for (actual, expected) in approx32.iter().zip(expected32.approximation()) {
        assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
    }

    let mut recovered32 = [0.0_f32; 8];
    plan.inverse_typed_into(
        &approx32,
        &details32,
        &mut recovered32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed f32 inverse");
    for (actual, expected) in recovered32.iter().zip(signal32.iter()) {
        assert!((*actual - *expected).abs() < 1.0e-5);
    }

    let signal16 = signal64.map(|value| f16::from_f32(value as f32));
    let represented16 = signal16.map(|value| f64::from(value.to_f32()));
    let expected16 = plan
        .forward(&represented16)
        .expect("represented f16 forward");
    let mut approx16 = vec![f16::from_f32(0.0); 1];
    let mut details16 = detail_buffers(&plan, f16::from_f32(0.0));
    plan.forward_typed_into(
        &signal16,
        &mut approx16,
        &mut details16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed f16 forward");
    for (actual, expected) in approx16.iter().zip(expected16.approximation()) {
        let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
    }
}

#[test]
fn leto_forward_and_inverse_match_slice_reference() {
    let plan = DwtPlan::new(8, 3, DiscreteWavelet::Haar).expect("valid DWT plan");
    let signal = [1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let leto_signal =
        leto::Array1::from_shape_vec([signal.len()], signal.to_vec()).expect("leto signal");
    let expected = plan.forward(&signal).expect("slice forward");

    let actual = plan.forward_leto(leto_signal.view()).expect("leto forward");
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.levels(), expected.levels());
    let actual_approximation = actual.approximation().view();
    let actual_approximation = actual_approximation
        .as_slice()
        .expect("contiguous approximation");
    for (actual, expected) in actual_approximation.iter().zip(expected.approximation()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
    for (actual_detail, expected_detail) in actual.details().iter().zip(expected.details()) {
        let actual_detail = actual_detail.view();
        let actual_detail = actual_detail.as_slice().expect("contiguous detail");
        for (actual, expected) in actual_detail.iter().zip(expected_detail) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    let expected_inverse = plan.inverse(&expected).expect("slice inverse");
    let actual_inverse = plan.inverse_leto(&actual).expect("leto inverse");
    let actual_inverse_view = actual_inverse.view();
    let actual_inverse = actual_inverse_view
        .as_slice()
        .expect("contiguous inverse signal");
    for (actual, expected) in actual_inverse.iter().zip(expected_inverse.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_strided_forward_matches_slice_reference() {
    let plan = DwtPlan::new(8, 3, DiscreteWavelet::Haar).expect("valid DWT plan");
    let signal = [1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let mut interleaved = Vec::with_capacity(signal.len() * 2);
    for value in signal {
        interleaved.push(value);
        interleaved.push(99.0);
    }
    let leto_signal =
        leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto signal");
    let strided = leto_signal
        .view()
        .slice(&[(0, signal.len() * 2, 2)])
        .expect("strided signal");
    let expected = plan.forward(&signal).expect("slice forward");

    let actual = plan.forward_leto(strided).expect("leto forward");
    let actual_approximation = actual.approximation().view();
    let actual_approximation = actual_approximation
        .as_slice()
        .expect("contiguous approximation");
    for (actual, expected) in actual_approximation.iter().zip(expected.approximation()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
    for (actual_detail, expected_detail) in actual.details().iter().zip(expected.details()) {
        let actual_detail = actual_detail.view();
        let actual_detail = actual_detail.as_slice().expect("contiguous detail");
        for (actual, expected) in actual_detail.iter().zip(expected_detail) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_slice_reference() {
    let plan = DwtPlan::new(8, 3, DiscreteWavelet::Haar).expect("valid DWT plan");
    let signal = [1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let leto_signal =
        leto::Array1::from_shape_vec([signal.len()], signal.to_vec()).expect("leto signal");
    let mut expected_approximation = vec![0.0_f32; 1];
    let mut expected_details = detail_buffers(&plan, 0.0_f32);
    plan.forward_typed_into(
        &signal,
        &mut expected_approximation,
        &mut expected_details,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed slice forward");

    let actual = plan
        .forward_leto_typed(leto_signal.view(), PrecisionProfile::LOW_PRECISION_F32)
        .expect("typed leto forward");
    let actual_approximation = actual.approximation().view();
    let actual_approximation = actual_approximation
        .as_slice()
        .expect("contiguous approximation");
    for (actual, expected) in actual_approximation
        .iter()
        .zip(expected_approximation.iter())
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for (actual_detail, expected_detail) in actual.details().iter().zip(expected_details.iter()) {
        let actual_detail = actual_detail.view();
        let actual_detail = actual_detail.as_slice().expect("contiguous detail");
        for (actual, expected) in actual_detail.iter().zip(expected_detail) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    let mut expected_inverse = [0.0_f32; 8];
    plan.inverse_typed_into(
        &expected_approximation,
        &expected_details,
        &mut expected_inverse,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed slice inverse");
    let actual_inverse = plan
        .inverse_leto_typed(&actual, PrecisionProfile::LOW_PRECISION_F32)
        .expect("typed leto inverse");
    let actual_inverse_view = actual_inverse.view();
    let actual_inverse = actual_inverse_view
        .as_slice()
        .expect("contiguous inverse signal");
    for (actual, expected) in actual_inverse.iter().zip(expected_inverse.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn typed_dwt_rejects_profile_and_shape_mismatch() {
    let plan = DwtPlan::new(4, 2, DiscreteWavelet::Haar).expect("valid DWT plan");
    let signal = [1.0_f32, 2.0, 3.0, 4.0];
    let mut approximation = vec![0.0_f32; 1];
    let mut details = detail_buffers(&plan, 0.0_f32);
    assert!(matches!(
        plan.forward_typed_into(
            &signal,
            &mut approximation,
            &mut details,
            PrecisionProfile::HIGH_ACCURACY_F64
        ),
        Err(WaveletError::PrecisionMismatch)
    ));

    details[0].pop();
    assert!(matches!(
        plan.forward_typed_into(
            &signal,
            &mut approximation,
            &mut details,
            PrecisionProfile::LOW_PRECISION_F32
        ),
        Err(WaveletError::CoefficientShapeMismatch)
    ));
}
