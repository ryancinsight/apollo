//! Direct-sum tone, rejection and kernel regression cases.

use core::num::NonZeroUsize;
use std::f64::consts::{PI, TAU};

use eunomia::{Complex, Complex64, RealField};

use super::{argument, estimate_peaks, real_spectrum, summed, Frame, Model, Tone, LEN};
use crate::domain::contracts::error::PeakEstimationError;

const PHASES: [f64; 8] = [-3.1, -2.618, -2.5307, -1.4, -0.2, 0.7, 1.9, 3.1];

/// Lone real tones across the phase circle, near DC and Nyquist where the image
/// is strongest and on an integer bin, each read at its own bin and again at
/// its mirror.
fn single_tones_within_their_bounds<T: RealField>() {
    let model = Model::new(LEN);
    for (position, amplitude) in [
        (1.7, 0.6),
        (10.3, 1.0),
        (31.0, 0.25),
        (40.00003, 1.0),
        (47.49, 3.0),
        (62.2, 1.5),
    ] {
        for phase in PHASES {
            let tone = Tone {
                position,
                amplitude,
                phase,
            };
            let spectrum = real_spectrum::<T>(&[tone]);
            let round = model.rounding::<T>(amplitude, argument(&[tone]));
            for tone in [tone, tone.mirrored(LEN)] {
                let k = tone.position.round() as usize;
                let estimates = estimate_peaks(&spectrum, &[k], NonZeroUsize::MIN)
                    .expect("invariant: a 128-bin frame and an in-range bin are valid");
                model.check::<T>(estimates[0], &tone, k, [round; 3], "single tone");
            }
        }
    }
}

#[test]
fn single_tones_within_their_bounds_for_every_scalar() {
    single_tones_within_their_bounds::<f64>();
    single_tones_within_their_bounds::<f32>();
}

/// One complex exponential has no image, so the inverse-tangent closure's
/// offset error is rounding alone. Placing it just below `N` makes the estimator
/// read bin `0` as the upper neighbour of bin `N − 1`. Without the correction
/// the offset errs by about `|δ| (1 − δ²) (π/N)² / 3`, outside this bound
/// in f64. The derivative of atan is at most one, so the ratio perturbation
/// propagates with at most `tan(π/N)/(π/N)`.
fn complex_exponential_offset_inverts_the_ratio<T: RealField>() {
    let model = Model::new(LEN);
    let n = LEN as f64;
    let (position, phase) = (127.4, 0.9);
    let samples: Vec<Complex64> = (0..LEN)
        .map(|t| Complex64::from_polar(1.0, TAU * position * t as f64 / n + phase))
        .collect();
    let spectrum = summed::<T>(&samples);
    let estimates = estimate_peaks(&spectrum, &[127], NonZeroUsize::MIN)
        .expect("invariant: a 128-bin frame and an in-range bin are valid");
    let estimate = estimates[0].expect("the exponential's bin holds its peak");
    let unit = Complex64::from_polar(1.0, phase);
    let term = |m: f64| unit * model.kernel(position - (127.0 + m));
    let denominator = term(0.0) * 2.0 - term(-1.0) - term(1.0);
    let exact_ratio = (term(-1.0) - term(1.0)) / denominator;
    let round = model.rounding::<T>(1.0, TAU * position + phase);
    let ratio = (2.0 * round + exact_ratio.norm() * 4.0 * round)
        / (denominator.norm() * (1.0 - 4.0 * round / denominator.norm()));
    let correction = (PI / n).tan() / (PI / n);
    let eps = T::EPSILON.to_f64();
    let bound = correction * ratio + 8.0 * eps + n * eps;
    let error = (estimate.position().to_f64() - position).abs();
    assert!(error <= bound, "offset error {error:e} exceeds {bound:e}");
}

#[test]
fn complex_exponential_offset_inverts_the_ratio_for_every_scalar() {
    complex_exponential_offset_inverts_the_ratio::<f64>();
    complex_exponential_offset_inverts_the_ratio::<f32>();
}

/// Two tones eight bins apart, the weaker at half the stronger, over three
/// rounds: the first read of the stronger tone carries the weaker one whole
/// (the direct pass), and every later read carries only the residual of the
/// estimate it subtracted.
fn separated_tones_resolve_after_peeling<T: RealField>() {
    let model = Model::new(LEN);
    for (strong_phase, weak_phase) in [(1.2, -0.6), (-2.9, 2.5), (0.4, 0.1)] {
        let tones = [
            Tone {
                position: 20.35,
                amplitude: 1.0,
                phase: strong_phase,
            },
            Tone {
                position: 28.1,
                amplitude: 0.5,
                phase: weak_phase,
            },
        ];
        let spectrum = real_spectrum::<T>(&tones);
        let round = model.rounding::<T>(1.5, argument(&tones));
        model.check_peeled::<T>(&spectrum, &tones, &[20, 28], 3, &|_| round, "peeled pair");
    }
}

#[test]
fn separated_tones_resolve_after_peeling_for_every_scalar() {
    separated_tones_resolve_after_peeling::<f64>();
    separated_tones_resolve_after_peeling::<f32>();
}

/// Bins that hold no peak of their own are rejected, not misread: a bin two
/// away from a lone tone reads an offset near two bins, and a tone read at DC
/// or Nyquist is its own image. For a real signal the offset read there is
/// exactly zero, the spectrum being conjugate-symmetric, and so is the
/// determinant; a complex exponential near Nyquist reads a non-zero offset,
/// and the estimator refuses DC and even-length Nyquist by index before its
/// solve.
fn bins_without_a_peak_are_rejected<T: RealField>() {
    let tone = Tone {
        position: 10.3,
        amplitude: 1.0,
        phase: 0.7,
    };
    let spectrum = real_spectrum::<T>(&[tone]);
    let estimates = estimate_peaks(&spectrum, &[12, 8], NonZeroUsize::MIN)
        .expect("invariant: in-range bins are valid");
    assert_eq!(estimates, vec![None, None], "bins two away from the tone");

    for (label, position, bin) in [
        ("DC", 0.0, 0),
        ("near DC", 0.3, 0),
        ("Nyquist", 64.0, 64),
        ("near Nyquist", 64.2, 64),
        ("near Nyquist, below", 63.7, 64),
    ] {
        for phase in PHASES {
            let spectrum = real_spectrum::<T>(&[Tone {
                position,
                amplitude: 1.0,
                phase,
            }]);
            let estimates = estimate_peaks(&spectrum, &[bin], NonZeroUsize::MIN)
                .expect("invariant: in-range bins are valid");
            assert_eq!(
                estimates,
                vec![None],
                "a tone {label} read at bin {bin} is its own image (phase {phase})"
            );
        }
    }

    let n = LEN as f64;
    for position in [64.2, 63.7, 64.45] {
        for phase in PHASES {
            let samples: Vec<Complex64> = (0..LEN)
                .map(|t| Complex64::from_polar(1.0, TAU * position * t as f64 / n + phase))
                .collect();
            let spectrum = summed::<T>(&samples);
            let estimates = estimate_peaks(&spectrum, &[LEN / 2], NonZeroUsize::MIN)
                .expect("invariant: in-range bins are valid");
            assert_eq!(
                estimates,
                vec![None],
                "an exponential at {position} read at Nyquist has a singular solve (phase {phase})"
            );
        }
    }
}

#[test]
fn bins_without_a_peak_are_rejected_for_every_scalar() {
    bins_without_a_peak_are_rejected::<f64>();
    bins_without_a_peak_are_rejected::<f32>();
}

#[test]
fn unreadable_inputs_are_typed_errors() {
    let short = vec![Complex::new(1.0_f64, 0.0); 2];
    assert_eq!(
        estimate_peaks(&short, &[0], NonZeroUsize::MIN),
        Err(PeakEstimationError::FrameTooShort { len: 2 })
    );
    let tone = Tone {
        position: 10.3,
        amplitude: 1.0,
        phase: 0.0,
    };
    let spectrum = real_spectrum::<f64>(&[tone]);
    assert_eq!(
        estimate_peaks(&spectrum, &[10, LEN], NonZeroUsize::MIN),
        Err(PeakEstimationError::PeakOutOfRange { bin: LEN, len: LEN })
    );
    assert_eq!(
        estimate_peaks(&spectrum, &[10, 30, 10], NonZeroUsize::MIN),
        Err(PeakEstimationError::DuplicatePeak { bin: 10 })
    );
    assert_eq!(
        estimate_peaks(&spectrum, &[10, 30, LEN - 10], NonZeroUsize::MIN),
        Err(PeakEstimationError::MirrorPeak {
            bin: LEN - 10,
            mirror: 10
        })
    );
    // DC and Nyquist are their own mirrors and stay readable.
    let readable = estimate_peaks(&spectrum, &[0, LEN / 2, 10], NonZeroUsize::MIN)
        .expect("self-mirrored bins are not a mirror pair");
    let model = Model::new(LEN);
    let round = model.rounding::<f64>(tone.amplitude, argument(&[tone]));
    model.check(readable[2], &tone, 10, [round; 3], "self-mirrored bins");
    // `N ε = ½` at `N = 2^22` in f32.
    let long = vec![Complex::new(0.0_f32, 0.0); 1 << 22];
    assert_eq!(
        estimate_peaks(&long, &[1], NonZeroUsize::MIN),
        Err(PeakEstimationError::FrameTooLong { len: 1 << 22 })
    );
}

/// The kernel against `Σ_{t<N} e^{2πi u t / N}` summed directly, at `u` as the
/// scalar holds it: near zero, near integers and multiples of `N` (where a
/// quotient form cancels), and thousands of bins away. The sum of `N` unit
/// phasors, each within `(2π + 3) ε`, carries at most `N (N + 2π + 3) ε`; the
/// product form a few `ε` relative to magnitudes up to `N`.
fn closed_form_kernel_matches_the_sum<T: RealField>() {
    let frame = Frame::<T>::new(LEN);
    let n = LEN as f64;
    let eps = T::EPSILON.to_f64();
    for u in [
        0.0, 3.1e-5, -0.37, 0.63, 1.00003, 3.5, 49.23, 127.9999, 128.00002, -98.63, 1_279.5,
        -17_802.37,
    ] {
        let held = T::from_f64(u);
        let closed = frame.kernel(held);
        let u = held.to_f64();
        let whole = u.round();
        let summed = (0..LEN).fold(Complex::new(T::ZERO, T::ZERO), |sum, t| {
            let turns = (u - whole) * t as f64 / n + (whole * t as f64).rem_euclid(n) / n;
            let angle = TAU * turns;
            sum + Complex::new(T::from_f64(angle.cos()), T::from_f64(angle.sin()))
        });
        let bound = n * (n + TAU + 3.0) * eps + 16.0 * eps * summed.norm().to_f64();
        let error = (closed - summed).norm().to_f64();
        assert!(
            error <= bound,
            "kernel at {u}: {closed:?} against the sum {summed:?}, {error:e} > {bound:e}"
        );
    }
}

#[test]
fn closed_form_kernel_matches_the_sum_for_every_scalar() {
    closed_form_kernel_matches_the_sum::<f64>();
    closed_form_kernel_matches_the_sum::<f32>();
}
