//! 1-D STFT plan value tests.
//!
//! The leto-reference parity tests live in the `tests/leto_reference.rs`
//! sidecar; this file crossed the 500-line gate when the parity wave landed.

mod leto_reference;

use super::{
    inverse_real_lane_workspace_capacity, inverse_wola_workspace_capacities,
    typed_workspace_capacities, window_complex_real_frame_into, window_signal_frame_into, StftPlan,
    HERMES_WINDOW_FRAME_THRESHOLD,
};
use crate::application::execution::kernel::window::Window;
use crate::domain::contracts::error::StftError;
use apollo_fft::{PrecisionProfile, F16};
use eunomia::assert_relative_eq;
use eunomia::{Complex32, Complex64};
use leto::Array1;
use proptest::prelude::*;

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
fn interior_frame_windowing_is_fused_and_bit_exact_with_the_scalar_formula() {
    // A frame wholly inside the signal takes the provider's fused
    // multiply-and-interleave straight from the signal slice. The product is
    // the same single rounding the scalar formula performs, so the comparison
    // is bit-exact, and the imaginary lane must be exactly zero.
    let len = HERMES_WINDOW_FRAME_THRESHOLD * 4;
    let signal: Vec<f64> = (0..len * 3).map(|i| (i as f64 * 0.17).cos()).collect();
    let window: Vec<f64> = (0..len)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (len - 1) as f64).cos())
        .collect();
    let start = (len / 3) as isize;
    let mut actual = vec![Complex64::new(7.0, 7.0); len];
    window_signal_frame_into(start, &signal, &window, &mut actual);
    for (n, actual) in actual.iter().enumerate() {
        let expected = signal[start as usize + n] * window[n];
        assert_eq!(actual.re.to_bits(), expected.to_bits(), "lane {n}");
        assert_eq!(actual.im.to_bits(), 0.0f64.to_bits(), "lane {n}");
    }
}

#[test]
fn forward_windowing_retains_no_real_scratch() {
    // Both forward real pools are gone: after interior and overhanging frames
    // at a provider-sized length, the only remaining real-lane pool -- the
    // inverse path's -- has never grown on this thread.
    let len = HERMES_WINDOW_FRAME_THRESHOLD * 8;
    let signal: Vec<f64> = (0..len * 2).map(|i| (i as f64 * 0.05).sin()).collect();
    let window = vec![0.75; len];
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    window_signal_frame_into(0, &signal, &window, &mut out);
    window_signal_frame_into(-(len as isize / 2), &signal, &window, &mut out);
    window_signal_frame_into((len + len / 2) as isize, &signal, &window, &mut out);
    assert_eq!(
        inverse_real_lane_workspace_capacity(),
        0,
        "forward windowing must not touch the inverse real-lane pool"
    );
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
fn window_values_are_validated() {
    assert_eq!(
        StftPlan::with_window_values(8, 4, vec![1.0f64; 6]).err(),
        Some(StftError::WindowLengthMismatch)
    );
    let mut values = vec![1.0f64; 8];
    values[3] = f64::NAN;
    assert_eq!(
        StftPlan::with_window_values(8, 4, values).err(),
        Some(StftError::InvalidWindowParameter)
    );
}

#[test]
fn hann_plan_is_the_default_plan() {
    let default = StftPlan::new(8, 4).expect("valid plan");
    let hann = StftPlan::with_window(8, 4, Window::Hann).expect("valid plan");
    assert_eq!(default.window(), hann.window());
}

/// Every window family, and a caller-supplied window, reconstructs what the
/// same plan analyzed, at half and quarter hops and at every sample, ends
/// included. The capability audit measured 4.1e-2 for a Hamming analysis
/// inverted with Hann synthesis.
///
/// Bound, per sample: the inverse divides `Σ_m w_m y_m` by `W = Σ_m w_m²`,
/// where `y_m` is frame `m` after a forward and an inverse FFT, within
/// `γ ‖w x‖₂ ≤ γ √N |w|_max |x|_max` per sample with `γ = 16 ⌈log₂ N⌉ ε`
/// (twice the `≈ 5.7 log₂N ε` of Higham, Accuracy and Stability of
/// Numerical Algorithms, 2nd ed., Theorem 24.2, per transform, with margin).
/// At most `K = ⌈N / hop⌉` frames contribute, so the numerator is within
/// `K |w|_max² γ √N |x|_max` of `W x`; its `K` products and sums and `W`'s
/// own add `(2K + 2) ε W |x|_max`, and the division one more `ε`. Dividing
/// by `W` gives `K |w|_max² γ √N |x|_max / W + (2K + 3) ε |x|_max`, which
/// scales with the window only through `|w|_max² / W`, so a scaled window
/// keeps its bound.
#[test]
fn every_window_reconstructs_what_it_analyzed() {
    let frame_len = 64usize;
    let signal_len = 512usize;
    let signal = Array1::from(
        (0..signal_len)
            .map(|i| (i as f64 * 0.37).sin() + 0.5 * (i as f64 * 0.113).cos())
            .collect::<Vec<_>>(),
    );
    let peak = signal.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let ramp: Vec<f64> = (0..frame_len).map(|i| 0.2 + (i as f64 / 63.0)).collect();
    let scaled_hann: Vec<f64> = Window::Hann
        .coefficients(frame_len)
        .expect("valid")
        .iter()
        .map(|w| 1.0e6 * w)
        .collect();
    for hop_len in [frame_len / 2, frame_len / 4] {
        let plans = [
            StftPlan::with_window(frame_len, hop_len, Window::Hann),
            StftPlan::with_window(frame_len, hop_len, Window::Hamming),
            StftPlan::with_window(frame_len, hop_len, Window::Blackman),
            StftPlan::with_window(frame_len, hop_len, Window::Tukey { alpha: 0.5 }),
            StftPlan::with_window_values(frame_len, hop_len, ramp.clone()),
            StftPlan::with_window_values(frame_len, hop_len, scaled_hann.clone()),
        ];
        for plan in plans {
            let plan = plan.expect("valid plan");
            let window = plan.window().as_slice().expect("contiguous").to_vec();
            let largest = window.iter().fold(0.0f64, |m, w| m.max(w.abs()));
            let frames = frame_len.div_ceil(hop_len) as f64;
            let gamma = 16.0 * (frame_len as f64).log2().ceil() * f64::EPSILON;
            let spectrum = plan.forward(&signal).expect("forward");
            let recovered = plan.inverse(&spectrum, signal_len).expect("inverse");
            let half = frame_len / 2;
            let last_frame = signal_len.div_ceil(hop_len);
            for i in 0..signal_len {
                let lowest = (i + half + 1).saturating_sub(frame_len).div_ceil(hop_len);
                let highest = ((i + half) / hop_len).min(last_frame);
                let weight: f64 = (lowest..=highest)
                    .map(|m| window[i + half - m * hop_len].powi(2))
                    .sum();
                let bound = frames * largest * largest * gamma * (frame_len as f64).sqrt() * peak
                    / weight
                    + (2.0 * frames + 3.0) * f64::EPSILON * peak;
                let error = (recovered[i] - signal[i]).abs();
                assert!(
                    error <= bound,
                    "hop {hop_len} window[0..2] {:?}: sample {i} error {error:e} > {bound:e}",
                    &window[..2]
                );
            }
        }
    }
}

/// The review's end-of-signal reproduction: every residue has energy, but
/// the first samples are covered only by the window's zero half.
#[test]
fn uncovered_signal_ends_refuse_the_inverse() {
    let plan = StftPlan::with_window_values(8, 2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        .expect("valid plan");
    let signal = Array1::from(
        (0..16)
            .map(|i| (i as f64 * 0.3).sin() + 0.6)
            .collect::<Vec<_>>(),
    );
    let spectrum = plan.forward(&signal).expect("the forward is defined");
    assert_eq!(
        plan.inverse(&spectrum, 16).err(),
        Some(StftError::WindowNotOverlapAdd)
    );
}

#[test]
fn a_window_that_does_not_overlap_add_refuses_the_inverse() {
    // Symmetric Hann is zero at both ends: a hop of the whole frame gives
    // residue 0 no energy.
    let plan = StftPlan::new(8, 8).expect("valid plan");
    let signal = Array1::from((0..32).map(|i| (i as f64 * 0.3).sin()).collect::<Vec<_>>());
    let spectrum = plan.forward(&signal).expect("the forward is defined");
    assert_eq!(
        plan.inverse(&spectrum, 32).err(),
        Some(StftError::WindowNotOverlapAdd)
    );
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
