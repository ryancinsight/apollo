use super::{inverse_wola_workspace_capacities, typed_workspace_capacities, StftPlan};
use crate::application::execution::kernel::hann::hann_window;
use crate::domain::contracts::error::StftError;
use apollo_fft::{f16, PrecisionProfile};
use approx::assert_relative_eq;
use ndarray::Array1;
use num_complex::{Complex32, Complex64};
use proptest::prelude::*;

#[test]
fn hann_window_is_symmetric() {
    let window = hann_window(8);
    for i in 0..8 {
        assert_relative_eq!(window[i], window[7 - i], epsilon = 1.0e-12);
    }
}

#[test]
fn forward_and_inverse_roundtrip_for_cola_case() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec(vec![
        1.0, -1.0, 0.5, 2.0, -0.75, 0.25, 1.5, -0.5, 0.125, 0.875, -1.25, 0.75,
    ]);
    let spectrum = plan.forward(&signal).expect("forward");
    let recovered = plan.inverse(&spectrum, signal.len()).expect("inverse");
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1.0e-8);
    }
}

#[test]
fn forward_into_matches_allocating_path() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec((0..16).map(|i| (i as f64 * 0.2).sin()).collect());
    let expected = plan.forward(&signal).expect("forward");
    let mut actual = Array1::<Complex64>::zeros(expected.len());
    plan.forward_into(&signal, &mut actual)
        .expect("forward_into");
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-12);
        assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_forward_matches_ndarray_reference() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec((0..16).map(|i| (i as f64 * 0.2).sin()).collect());
    let leto_signal =
        leto::Array1::from_shape_vec([signal.len()], signal.iter().copied().collect())
            .expect("leto signal");
    let expected = plan.forward(&signal).expect("ndarray forward");

    let actual = plan.forward_leto(leto_signal.view()).expect("leto forward");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous leto output");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_strided_forward_matches_ndarray_reference() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec((0..16).map(|i| (i as f64 * 0.2).sin()).collect());
    let mut interleaved = Vec::with_capacity(signal.len() * 2);
    for value in signal.iter().copied() {
        interleaved.push(value);
        interleaved.push(99.0);
    }
    let leto_signal =
        leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto signal");
    let strided = leto_signal
        .view()
        .slice(&[(0, signal.len() * 2, 2)])
        .expect("strided signal");
    let expected = plan.forward(&signal).expect("ndarray forward");

    let actual = plan.forward_leto(strided).expect("leto forward");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous leto output");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_inverse_matches_ndarray_reference() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec((0..16).map(|i| (i as f64 * 0.2).sin()).collect());
    let spectrum = plan.forward(&signal).expect("ndarray forward");
    let leto_spectrum =
        leto::Array1::from_shape_vec([spectrum.len()], spectrum.iter().copied().collect())
            .expect("leto spectrum");
    let expected = plan
        .inverse(&spectrum, signal.len())
        .expect("ndarray inverse");

    let actual = plan
        .inverse_leto(leto_spectrum.view(), signal.len())
        .expect("leto inverse");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous leto output");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn inverse_into_reuses_wola_workspace_capacity() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec((0..16).map(|i| (i as f64 * 0.2).sin()).collect());
    let spectrum = plan.forward(&signal).expect("forward");
    let frame_work_len = plan.frame_count(signal.len()) * plan.frame_len();
    let mut first = Array1::<f64>::zeros(signal.len());
    let mut second = Array1::<f64>::zeros(signal.len());

    plan.inverse_into(&spectrum, signal.len(), &mut first)
        .expect("first inverse");
    let after_first = inverse_wola_workspace_capacities();
    assert!(after_first.0 >= frame_work_len);
    assert!(after_first.1 >= frame_work_len);
    assert!(after_first.2 >= signal.len());
    assert!(after_first.3 >= signal.len());

    plan.inverse_into(&spectrum, signal.len(), &mut second)
        .expect("second inverse");
    assert_eq!(inverse_wola_workspace_capacities(), after_first);
    for ((lhs, rhs), expected) in first.iter().zip(second.iter()).zip(signal.iter()) {
        assert_eq!(lhs.to_bits(), rhs.to_bits());
        assert_relative_eq!(lhs, expected, epsilon = 1.0e-8);
    }
}

#[test]
fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal64 = Array1::from_vec((0..16).map(|i| (i as f64 * 0.2).sin()).collect());
    let expected = plan.forward(&signal64).expect("forward");

    let mut out64 = Array1::<Complex64>::zeros(expected.len());
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
    let mut out32 = Array1::<Complex32>::zeros(expected32.len());
    plan.forward_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
        .expect("typed f32 forward");
    for (actual, expected) in out32.iter().zip(expected32.iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
        assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
    }

    let mut recovered32 = Array1::<f32>::zeros(signal32.len());
    plan.inverse_typed_into(
        &out32,
        signal32.len(),
        &mut recovered32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed f32 inverse");
    for (actual, expected) in recovered32.iter().zip(signal32.iter()) {
        assert!((*actual - *expected).abs() < 1.0e-4);
    }

    let signal16 = signal64.mapv(|value| f16::from_f32(value as f32));
    let represented16 = signal16.mapv(|value| f64::from(value.to_f32()));
    let expected16 = plan
        .forward(&represented16)
        .expect("represented f16 forward");
    let mut out16 = Array1::from_elem(expected16.len(), [f16::from_f32(0.0); 2]);
    plan.forward_typed_into(
        &signal16,
        &mut out16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed f16 forward");
    for (actual, expected) in out16.iter().zip(expected16.iter()) {
        let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
        assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_ndarray_reference() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec((0..16).map(|i| (i as f32 * 0.2).sin()).collect());
    let leto_signal =
        leto::Array1::from_shape_vec([signal.len()], signal.iter().copied().collect())
            .expect("leto signal");
    let spectrum_len = plan.frame_count(signal.len()) * plan.spectrum_len();
    let mut expected_spectrum = Array1::<Complex32>::zeros(spectrum_len);
    plan.forward_typed_into(
        &signal,
        &mut expected_spectrum,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed ndarray forward");

    let actual_spectrum = plan
        .forward_leto_typed::<f32, Complex32>(
            leto_signal.view(),
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed leto forward");
    let actual_spectrum_view = actual_spectrum.view();
    let actual_spectrum_slice = actual_spectrum_view
        .as_slice()
        .expect("contiguous leto output");
    for (actual, expected) in actual_spectrum_slice.iter().zip(expected_spectrum.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }

    let leto_spectrum =
        leto::Array1::from_shape_vec([expected_spectrum.len()], expected_spectrum.to_vec())
            .expect("leto spectrum");
    let mut expected_signal = Array1::<f32>::zeros(signal.len());
    plan.inverse_typed_into(
        &expected_spectrum,
        signal.len(),
        &mut expected_signal,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed ndarray inverse");

    let actual_signal = plan
        .inverse_leto_typed::<Complex32, f32>(
            leto_spectrum.view(),
            signal.len(),
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed leto inverse");
    let actual_signal_view = actual_signal.view();
    let actual_signal_slice = actual_signal_view
        .as_slice()
        .expect("contiguous leto output");
    for (actual, expected) in actual_signal_slice.iter().zip(expected_signal.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn typed_paths_reuse_bridge_workspace_capacity() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec((0..16).map(|i| (i as f32 * 0.2).sin()).collect());
    let spectrum_len = plan.frame_count(signal.len()) * plan.spectrum_len();
    let mut first_spectrum = Array1::<Complex32>::zeros(spectrum_len);
    let mut second_spectrum = Array1::<Complex32>::zeros(spectrum_len);

    plan.forward_typed_into(
        &signal,
        &mut first_spectrum,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("first typed forward");
    let after_first_forward = typed_workspace_capacities();
    assert!(after_first_forward.0 >= signal.len());
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

    let mut first_recovered = Array1::<f32>::zeros(signal.len());
    let mut second_recovered = Array1::<f32>::zeros(signal.len());
    plan.inverse_typed_into(
        &first_spectrum,
        signal.len(),
        &mut first_recovered,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("first typed inverse");
    let after_first_inverse = typed_workspace_capacities();
    assert!(after_first_inverse.1 >= spectrum_len);
    assert!(after_first_inverse.3 >= signal.len());

    plan.inverse_typed_into(
        &first_spectrum,
        signal.len(),
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
    let signal = Array1::from_vec(vec![1.0_f32; 16]);
    let mut output =
        Array1::<Complex32>::zeros(plan.frame_count(signal.len()) * plan.spectrum_len());
    assert!(matches!(
        plan.forward_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
        Err(StftError::PrecisionMismatch)
    ));
}

#[test]
fn rejects_invalid_parameters() {
    assert!(matches!(
        StftPlan::new(0, 4),
        Err(StftError::EmptyFrameLength)
    ));
    assert!(matches!(StftPlan::new(8, 0), Err(StftError::EmptyHopSize)));
    assert!(matches!(
        StftPlan::new(4, 8),
        Err(StftError::HopExceedsFrame)
    ));
}

#[test]
fn input_too_short_is_rejected() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec(vec![0.0; 4]);
    assert!(matches!(
        plan.forward(&signal),
        Err(StftError::InputTooShort)
    ));
}

#[test]
fn forward_with_window_rejects_wrong_length() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec(vec![1.0f64; 12]);
    let bad_window = vec![1.0f64; 6];
    assert!(matches!(
        plan.forward_with_window(&signal, &bad_window),
        Err(StftError::WindowLengthMismatch)
    ));
}

#[test]
fn forward_with_custom_window_matches_internal_hann() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from_vec((0..12).map(|i| (i as f64 * 0.3).sin()).collect());
    let expected = plan.forward(&signal).expect("forward");
    let window: Vec<f64> = hann_window(8).to_vec();
    let actual = plan
        .forward_with_window(&signal, &window)
        .expect("forward_with_window");
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-12);
        assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-12);
    }
}

proptest::proptest! {
    #[test]
    fn roundtrip_holds_for_random_signals(
        signal_len in 8usize..128,
        frame_len in 2usize..17,
        hop_len in 1usize..9,
    ) {
        prop_assume!(frame_len <= signal_len);
        prop_assume!(hop_len <= frame_len);
        prop_assume!(hop_len + 2 <= frame_len);
        let plan = StftPlan::new(frame_len, hop_len).expect("valid plan");
        let signal = Array1::from_vec(
            (0..signal_len).map(|i| (i as f64 * 0.37).sin()).collect(),
        );
        let spectrum = plan.forward(&signal).expect("forward");
        let recovered = plan.inverse(&spectrum, signal_len).expect("inverse");
        let err = signal
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        prop_assert!(err < 0.5, "roundtrip error too large: {}", err);
    }
}
