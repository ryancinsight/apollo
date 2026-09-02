//! Leto-reference parity tests for the 1-D STFT plan.
//!
//! Sidecar of `tests.rs` (which crossed the 500-line gate); these tests
//! compare the plan's leto-bridge paths against a `leto::Array1` reference.

use super::StftPlan;
use apollo_fft::PrecisionProfile;
use eunomia::assert_relative_eq;
use eunomia::Complex32;
use leto::Array1;

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
