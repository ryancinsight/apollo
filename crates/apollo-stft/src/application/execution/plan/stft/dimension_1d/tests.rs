use super::{
    forward_window_workspace_capacity, inverse_wola_workspace_capacities,
    typed_workspace_capacities, window_complex_real_frame_into, window_signal_frame_into, StftPlan,
    HERMES_WINDOW_FRAME_THRESHOLD,
};
use crate::application::execution::kernel::hann::hann_window;
use crate::domain::contracts::error::StftError;
use apollo_fft::{f16, PrecisionProfile};
use approx::assert_relative_eq;
use eunomia::{Complex32, Complex64};
use leto::Array1;
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
    let signal = Array1::from(vec![
        1.0, -1.0, 0.5, 2.0, -0.75, 0.25, 1.5, -0.5, 0.125, 0.875, -1.25, 0.75,
    ]);
    let spectrum = plan.forward(&signal).expect("forward");
    let recovered = plan.inverse(&spectrum, signal.size()).expect("inverse");
    for (actual, expected) in recovered.iter().zip(signal.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1.0e-8);
    }
}

#[test]
fn forward_into_matches_allocating_path() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from((0..16).map(|i| (i as f64 * 0.2).sin()).collect::<Vec<_>>());
    let expected = plan.forward(&signal).expect("forward");
    let mut actual = Array1::<Complex64>::zeros([expected.size()]);
    plan.forward_into(&signal, &mut actual)
        .expect("forward_into");
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-12);
        assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_forward_matches_leto_reference() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from((0..16).map(|i| (i as f64 * 0.2).sin()).collect::<Vec<_>>());
    let leto_signal =
        leto::Array1::from_shape_vec([signal.size()], signal.iter().copied().collect::<Vec<_>>())
            .expect("leto signal");
    let expected = plan.forward(&signal).expect("leto forward");

    let actual = plan.forward_leto(leto_signal.view()).expect("leto forward");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous leto output");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_strided_forward_matches_leto_reference() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from((0..16).map(|i| (i as f64 * 0.2).sin()).collect::<Vec<_>>());
    let mut interleaved = Vec::with_capacity(signal.size() * 2);
    for value in signal.iter().copied() {
        interleaved.push(value);
        interleaved.push(99.0);
    }
    let leto_signal =
        leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto signal");
    let strided = leto_signal
        .view()
        .slice(&[(0, signal.size() * 2, 2)])
        .expect("strided signal");
    let expected = plan.forward(&signal).expect("leto forward");

    let actual = plan.forward_leto(strided).expect("leto forward");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous leto output");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_inverse_matches_leto_reference() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from((0..16).map(|i| (i as f64 * 0.2).sin()).collect::<Vec<_>>());
    let spectrum = plan.forward(&signal).expect("leto forward");
    let leto_spectrum = leto::Array1::from_shape_vec(
        [spectrum.size()],
        spectrum.iter().copied().collect::<Vec<_>>(),
    )
    .expect("leto spectrum");
    let expected = plan
        .inverse(&spectrum, signal.size())
        .expect("leto inverse");

    let actual = plan
        .inverse_leto(leto_spectrum.view(), signal.size())
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
    let signal = Array1::from((0..16).map(|i| (i as f64 * 0.2).sin()).collect::<Vec<_>>());
    let spectrum = plan.forward(&signal).expect("forward");
    let frame_work_len = plan.frame_count(signal.size()) * plan.frame_len();
    let mut first = Array1::<f64>::zeros([signal.size()]);
    let mut second = Array1::<f64>::zeros([signal.size()]);

    plan.inverse_into(&spectrum, signal.size(), &mut first)
        .expect("first inverse");
    let after_first = inverse_wola_workspace_capacities();
    assert!(after_first.0 >= frame_work_len);
    assert!(after_first.1 >= frame_work_len);
    assert!(after_first.2 >= signal.size());
    assert!(after_first.3 >= signal.size());

    plan.inverse_into(&spectrum, signal.size(), &mut second)
        .expect("second inverse");
    assert_eq!(inverse_wola_workspace_capacities(), after_first);
    for ((lhs, rhs), expected) in first.iter().zip(second.iter()).zip(signal.iter()) {
        assert_eq!(lhs.to_bits(), rhs.to_bits());
        assert_relative_eq!(lhs, expected, epsilon = 1.0e-8);
    }
}

#[test]
fn hermes_forward_windowing_matches_scalar_formula_at_threshold() {
    let signal: Vec<f64> = (0..(HERMES_WINDOW_FRAME_THRESHOLD + 16))
        .map(|i| (i as f64 * 0.17).cos())
        .collect();
    let window: Vec<f64> = (0..HERMES_WINDOW_FRAME_THRESHOLD)
        .map(|i| {
            0.5 - 0.5
                * (std::f64::consts::TAU * i as f64 / (HERMES_WINDOW_FRAME_THRESHOLD - 1) as f64)
                    .cos()
        })
        .collect();
    let start = -5;
    let mut actual = vec![Complex64::new(0.0, 0.0); HERMES_WINDOW_FRAME_THRESHOLD];

    window_signal_frame_into(start, &signal, &window, &mut actual);

    assert!(forward_window_workspace_capacity() >= HERMES_WINDOW_FRAME_THRESHOLD);
    for (n, actual) in actual.iter().enumerate() {
        let signal_index = start + n as isize;
        let expected = if signal_index >= 0 && (signal_index as usize) < signal.len() {
            signal[signal_index as usize] * window[n]
        } else {
            0.0
        };
        assert_relative_eq!(actual.re, expected, epsilon = 1.0e-12);
        assert_eq!(actual.im.to_bits(), 0.0f64.to_bits());
    }
}

#[test]
fn hermes_inverse_windowing_matches_scalar_formula_at_threshold() {
    let frame: Vec<Complex64> = (0..HERMES_WINDOW_FRAME_THRESHOLD)
        .map(|i| Complex64::new((i as f64 * 0.13).sin(), (i as f64 * 0.19).cos()))
        .collect();
    let window: Vec<f64> = (0..HERMES_WINDOW_FRAME_THRESHOLD)
        .map(|i| {
            0.5 - 0.5
                * (std::f64::consts::TAU * i as f64 / (HERMES_WINDOW_FRAME_THRESHOLD - 1) as f64)
                    .cos()
        })
        .collect();
    let mut actual = vec![0.0; HERMES_WINDOW_FRAME_THRESHOLD];

    window_complex_real_frame_into(&frame, &window, &mut actual);

    for ((actual, frame), window) in actual.iter().zip(frame.iter()).zip(window.iter()) {
        assert_relative_eq!(*actual, frame.re * window, epsilon = 1.0e-12);
    }
}

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

    let signal16 = signal64.mapv(|value| f16::from_f32(value as f32));
    let represented16 = signal16.mapv(|value| f64::from(value.to_f32()));
    let expected16 = plan
        .forward(&represented16)
        .expect("represented f16 forward");
    let mut out16 = Array1::from_elem([expected16.size()], [f16::from_f32(0.0); 2]);
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
fn typed_leto_forward_and_inverse_match_leto_reference() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from((0..16).map(|i| (i as f32 * 0.2).sin()).collect::<Vec<_>>());
    let leto_signal =
        leto::Array1::from_shape_vec([signal.size()], signal.iter().copied().collect::<Vec<_>>())
            .expect("leto signal");
    let spectrum_len = plan.frame_count(signal.size()) * plan.spectrum_len();
    let mut expected_spectrum = Array1::<Complex32>::zeros([spectrum_len]);
    plan.forward_typed_into(
        &signal,
        &mut expected_spectrum,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed leto forward");

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

    let leto_spectrum = leto::Array1::from_shape_vec(
        [expected_spectrum.size()],
        expected_spectrum
            .as_slice()
            .expect("contiguous leto output")
            .to_vec(),
    )
    .expect("leto spectrum");
    let mut expected_signal = Array1::<f32>::zeros([signal.size()]);
    plan.inverse_typed_into(
        &expected_spectrum,
        signal.size(),
        &mut expected_signal,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed leto inverse");

    let actual_signal = plan
        .inverse_leto_typed::<Complex32, f32>(
            leto_spectrum.view(),
            signal.size(),
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
    let signal = Array1::from(vec![0.0; 4]);
    assert!(matches!(
        plan.forward(&signal),
        Err(StftError::InputTooShort)
    ));
}

#[test]
fn forward_with_window_rejects_wrong_length() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from(vec![1.0f64; 12]);
    let bad_window = vec![1.0f64; 6];
    assert!(matches!(
        plan.forward_with_window(&signal, &bad_window),
        Err(StftError::WindowLengthMismatch)
    ));
}

#[test]
fn forward_with_custom_window_matches_internal_hann() {
    let plan = StftPlan::new(8, 4).expect("valid plan");
    let signal = Array1::from((0..12).map(|i| (i as f64 * 0.3).sin()).collect::<Vec<_>>());
    let expected = plan.forward(&signal).expect("forward");
    let window: Vec<f64> = hann_window(8).into_vec();
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
        let signal = Array1::from(
            (0..signal_len).map(|i| (i as f64 * 0.37).sin()).collect::<Vec<_>>(),
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
