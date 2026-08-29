//! Row-independent selected-axis FFT contracts.

use crate::infrastructure::transport::gpu::{FramePlan, FramedExecution, StftWgpuPlan};

use super::oracle::{forward, frame_count, inverse, operation_bound, spectrum};
use super::support::backend;

#[test]
fn selected_axis_forward_matches_every_direct_dft_bin() {
    let Some(backend) = backend() else {
        return;
    };
    for (frame_len, hop_len, signal_len) in [(8, 4, 17), (6, 3, 14)] {
        let signal = (0..signal_len)
            .map(|index| {
                let x = index as f32;
                (0.37 * x).sin() + 0.25 * (0.11 * x * x).cos() - 0.03 * x
            })
            .collect::<Vec<_>>();
        let plan = StftWgpuPlan::new(FramePlan::new(frame_len, hop_len));
        let actual = backend
            .execute_forward(&plan, &signal)
            .expect("selected-axis forward");
        let expected = forward(&signal, frame_len, hop_len);
        assert_eq!(actual.len(), expected.len());

        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let frame = index / frame_len;
            let start = frame * hop_len;
            let scale = signal
                .get(start.saturating_sub(frame_len / 2)..)
                .unwrap_or(&[])
                .iter()
                .take(frame_len)
                .map(|value| f64::from(value.abs()))
                .sum();
            let bound = operation_bound(frame_len, scale);
            let error =
                (f64::from(actual.re) - expected.re).hypot(f64::from(actual.im) - expected.im);
            assert!(
                error <= bound,
                "frame {frame}, bin {}: error {error:.3e} exceeds {bound:.3e}",
                index % frame_len
            );
        }
    }
}

#[test]
fn selected_axis_inverse_matches_direct_dft_and_normalization() {
    let Some(backend) = backend() else {
        return;
    };
    for (frame_len, hop_len, signal_len) in [(8, 4, 17), (6, 3, 14)] {
        let frames = frame_count(signal_len, hop_len);
        let input = spectrum(frames, frame_len, 2);
        let plan = StftWgpuPlan::new(FramePlan::new(frame_len, hop_len));
        let actual = backend
            .execute_inverse(&plan, &input, signal_len)
            .expect("selected-axis inverse");
        let expected = inverse(&input, signal_len, frame_len, hop_len);
        let scale = input
            .iter()
            .map(|value| f64::from(value.re).hypot(f64::from(value.im)))
            .sum::<f64>()
            / frames as f64;
        let bound = operation_bound(frame_len, scale);
        for (sample, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let error = (f64::from(*actual) - expected).abs();
            assert!(
                error <= bound,
                "sample {sample}: error {error:.3e} exceeds {bound:.3e}"
            );
        }

        let max_expected = expected.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let max_actual = actual
            .iter()
            .copied()
            .map(f64::from)
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(max_expected > 0.0, "normalization sentinel must be nonzero");
        assert!((max_actual - max_expected).abs() <= bound);
        assert!((max_actual - max_expected * frame_len as f64).abs() > bound);
        assert!((max_actual - max_expected / frame_len as f64).abs() > bound);
    }
}
