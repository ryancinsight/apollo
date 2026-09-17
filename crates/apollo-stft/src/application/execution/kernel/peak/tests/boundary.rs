//! Boundary classification against DFTs summed directly from time samples.
//!
//! The oracle does not call [`super::super::Frame::kernel`]. Its Dirichlet
//! kernel is the defining phasor sum. For a bin perturbation bounded by `E`,
//! the three-bin ratio has numerator error at most `2E` and denominator error
//! at most `4E`; the quotient perturbation below follows from the triangle
//! inequality. The inverse-tangent closure is
//! `atan(r tan(π/N)) / (π/N)`, whose Lipschitz factor is at most
//! `tan(π/N) / (π/N)`.

use core::num::NonZeroUsize;
use std::f64::consts::{PI, TAU};

use eunomia::{Complex, Complex64, RealField};

use super::{estimate_peaks, summed, Model, PeakEstimate, Tone};

fn kernel_sum(len: usize, u: f64) -> Complex64 {
    let n = len as f64;
    (0..len).fold(Complex64::new(0.0, 0.0), |sum, t| {
        sum + Complex64::from_polar(1.0, TAU * u * t as f64 / n)
    })
}

fn complex_spectrum<T: RealField>(len: usize, position: f64, phase: f64) -> Vec<Complex<T>> {
    let n = len as f64;
    let samples: Vec<Complex64> = (0..len)
        .map(|t| Complex64::from_polar(1.0, TAU * position * t as f64 / n + phase))
        .collect();
    summed(&samples)
}

fn real_spectrum<T: RealField>(len: usize, tone: &Tone) -> Vec<Complex<T>> {
    let n = len as f64;
    let samples: Vec<Complex64> = (0..len)
        .map(|t| {
            Complex64::new(
                tone.amplitude * (TAU * tone.position * t as f64 / n + tone.phase).cos(),
                0.0,
            )
        })
        .collect();
    summed(&samples)
}

fn rounding<T: RealField>(len: usize, amplitude: f64, position: f64, phase: f64) -> f64 {
    Model::new(len).rounding::<T>(amplitude, TAU * position.abs() + phase.abs())
}

fn close_ratio(len: usize, ratio: Complex64) -> f64 {
    let step = PI / len as f64;
    (ratio.re * step.tan()).atan() / step
}

fn ratio_bound(before: Complex64, peak: Complex64, after: Complex64, error: f64) -> f64 {
    let denominator = peak * 2.0 - before - after;
    let ratio = (before - after) / denominator;
    let denominator_error = 4.0 * error;
    assert!(
        denominator_error < denominator.norm(),
        "the selected boundary case must have a stable three-bin ratio"
    );
    (2.0 * error + ratio.norm() * denominator_error) / (denominator.norm() - denominator_error)
}

fn complex_offset_bound<T: RealField>(len: usize, position: f64, phase: f64, bin: usize) -> f64 {
    let unit = Complex64::from_polar(1.0, phase);
    let value = |p: f64| unit * kernel_sum(len, position - p);
    let before = value(bin as f64 - 1.0);
    let peak = value(bin as f64);
    let after = value(bin as f64 + 1.0);
    let ratio = (before - after) / (peak * 2.0 - before - after);
    let delta = position - bin as f64;
    let oracle_residual = (close_ratio(len, ratio) - delta).abs();
    let step = PI / len as f64;
    let propagated = step.tan() / step
        * ratio_bound(
            before,
            peak,
            after,
            rounding::<T>(len, 1.0, position, phase),
        );
    oracle_residual + propagated + (len as f64 + 32.0) * T::EPSILON.to_f64()
}

fn strict_complex_boundaries<T: RealField>() {
    for len in [16, 32, 64] {
        let bin = len / 4;
        for phase in [-2.4, 0.3, 2.7] {
            for direction in [-1.0, 1.0] {
                let inside = direction * 0.499_999;
                let position = bin as f64 + inside;
                let spectrum = complex_spectrum::<T>(len, position, phase);
                let estimate = estimate_peaks(&spectrum, &[bin], NonZeroUsize::MIN)
                    .expect("invariant: the direct DFT and its peak bin are valid")[0]
                    .unwrap_or_else(|| {
                        panic!(
                            "strictly interior offset {inside} was rejected for N={len}, phase={phase}"
                        )
                    });
                let bound = complex_offset_bound::<T>(len, position, phase, bin);
                let error = (estimate.position().to_f64() - position).abs();
                assert!(
                    error <= bound,
                    "N={len}, phase={phase}, offset={inside}: {error:e} > {bound:e}"
                );

                // Place the exterior case beyond the same derived rounding
                // bound. The factor four separates the two closed intervals.
                let guard = (4.0 * bound).max(1.0e-6);
                let outside = direction * (0.5 + guard);
                let position = bin as f64 + outside;
                let spectrum = complex_spectrum::<T>(len, position, phase);
                let estimates = estimate_peaks(&spectrum, &[bin], NonZeroUsize::MIN)
                    .expect("invariant: the direct DFT and its peak bin are valid");
                assert_eq!(
                    estimates,
                    vec![None],
                    "exterior offset {outside} was accepted for N={len}, phase={phase}"
                );
            }
        }
    }
}

#[test]
fn strict_complex_boundaries_are_classified_for_every_scalar() {
    strict_complex_boundaries::<f64>();
    strict_complex_boundaries::<f32>();
}

fn real_bin(tone: &Tone, len: usize, p: f64) -> Complex64 {
    let a = tone.a();
    a * kernel_sum(len, tone.position - p) + a.conj() * kernel_sum(len, -tone.position - p)
}

fn real_offset_bound<T: RealField>(tone: &Tone, len: usize, bin: usize) -> f64 {
    let before = real_bin(tone, len, bin as f64 - 1.0);
    let peak = real_bin(tone, len, bin as f64);
    let after = real_bin(tone, len, bin as f64 + 1.0);
    let ratio = (before - after) / (peak * 2.0 - before - after);
    let oracle_residual = (close_ratio(len, ratio) - (tone.position - bin as f64)).abs();
    let step = PI / len as f64;
    oracle_residual
        + step.tan() / step
            * ratio_bound(
                before,
                peak,
                after,
                rounding::<T>(len, tone.amplitude, tone.position, tone.phase),
            )
        + (len as f64 + 32.0) * T::EPSILON.to_f64()
}

fn phase_distance(left: f64, right: f64) -> f64 {
    (left - right + PI).rem_euclid(TAU) - PI
}

fn check_real_estimate<T: RealField>(
    estimate: PeakEstimate<T>,
    tone: &Tone,
    len: usize,
    bin: usize,
    label: &str,
) {
    let offset_bound = real_offset_bound::<T>(tone, len, bin);
    let position = estimate.position().to_f64();
    let position_error = (position - tone.position).abs();
    assert!(
        position_error <= offset_bound,
        "{label}: position error {position_error:e} > {offset_bound:e}"
    );

    let delta = tone.position - bin as f64;
    let estimated_delta = position - bin as f64;
    let direct = kernel_sum(len, estimated_delta);
    let image = kernel_sum(len, -(2.0 * bin as f64 + estimated_delta));
    let gap = direct.norm() - image.norm();
    assert!(gap > 0.0, "{label}: image solve has no magnitude gap");
    let moved = (kernel_sum(len, delta) - direct).norm()
        + (kernel_sum(len, -(2.0 * bin as f64 + delta)) - image).norm();
    let a = tone.amplitude / 2.0;
    let solve_bound = (a * moved + rounding::<T>(len, tone.amplitude, tone.position, tone.phase))
        / gap
        + 16.0 * T::EPSILON.to_f64() * a;

    let amplitude_error = (estimate.amplitude().to_f64() - tone.amplitude).abs();
    assert!(
        amplitude_error <= 2.0 * solve_bound,
        "{label}: amplitude error {amplitude_error:e} > {:e}",
        2.0 * solve_bound
    );
    assert!(
        solve_bound < a,
        "{label}: the phase bound requires a solve-error disk excluding the origin"
    );
    let phase_error = phase_distance(estimate.phase().to_f64(), tone.phase).abs();
    let phase_bound = (solve_bound / a).asin();
    assert!(
        phase_error <= phase_bound,
        "{label}: phase error {phase_error:e} > {phase_bound:e}"
    );
}

fn real_image_boundaries<T: RealField>() {
    for len in [32, 128] {
        let bin = len / 4;
        for phase in [-2.4, 0.3, 2.7] {
            for direction in [-1.0, 1.0] {
                let tone = Tone {
                    position: bin as f64 + direction * 0.49,
                    amplitude: 0.75,
                    phase,
                };
                let spectrum = real_spectrum::<T>(len, &tone);
                for (read_bin, expected, label) in [
                    (bin, tone, "direct image read"),
                    (len - bin, tone.mirrored(len), "mirrored image read"),
                ] {
                    let estimate = estimate_peaks(&spectrum, &[read_bin], NonZeroUsize::MIN)
                        .expect("invariant: the direct DFT and its peak bin are valid")[0]
                        .unwrap_or_else(|| {
                            panic!("{label} was rejected for N={len}, phase={phase}")
                        });
                    check_real_estimate(estimate, &expected, len, read_bin, label);
                }

                // A real tone's image biases the ratio. At 0.51 bins these
                // selected quarter-frame bins remain beyond the half-bin by
                // more than the independently evaluated image and rounding
                // bound, on both the direct and mirrored reads.
                let outside = Tone {
                    position: bin as f64 + direction * 0.51,
                    ..tone
                };
                let spectrum = real_spectrum::<T>(len, &outside);
                for read_bin in [bin, len - bin] {
                    let expected = if read_bin == bin {
                        outside
                    } else {
                        outside.mirrored(len)
                    };
                    let distance = (expected.position - read_bin as f64).abs() - 0.5;
                    let bound = real_offset_bound::<T>(&expected, len, read_bin);
                    assert!(
                        distance > bound,
                        "selected exterior case needs {distance:e} > {bound:e}"
                    );
                    let estimates = estimate_peaks(&spectrum, &[read_bin], NonZeroUsize::MIN)
                        .expect("invariant: the direct DFT and its peak bin are valid");
                    assert_eq!(estimates, vec![None], "exterior {expected:?} was accepted");
                }
            }
        }
    }
}

#[test]
fn real_image_boundaries_preserve_parameters_for_every_scalar() {
    real_image_boundaries::<f64>();
    real_image_boundaries::<f32>();
}

fn exact_half_bin_images<T: RealField>() {
    let mut accepted = 0;
    let mut rejected = 0;
    for len in [16, 32] {
        let bin = len / 4;
        for phase in [-2.4, 0.3, 2.7, PI / 2.0] {
            for direction in [-1.0, 1.0] {
                let tone = Tone {
                    position: bin as f64 + direction * 0.5,
                    amplitude: 0.75,
                    phase,
                };
                let before = real_bin(&tone, len, bin as f64 - 1.0);
                let peak = real_bin(&tone, len, bin as f64);
                let after = real_bin(&tone, len, bin as f64 + 1.0);
                let ratio = (before - after) / (peak * 2.0 - before - after);
                let reference = close_ratio(len, ratio);
                let step = PI / len as f64;
                let uncertainty = step.tan() / step
                    * ratio_bound(
                        before,
                        peak,
                        after,
                        rounding::<T>(len, tone.amplitude, tone.position, tone.phase),
                    )
                    + (len as f64 + 32.0) * T::EPSILON.to_f64();
                // The image, not the true offset alone, decides which side
                // of the estimated-offset boundary these cases occupy.
                assert!((reference.abs() - 0.5).abs() > uncertainty);
                let spectrum = real_spectrum::<T>(len, &tone);
                let actual = estimate_peaks(&spectrum, &[bin], NonZeroUsize::MIN)
                    .expect("invariant: direct spectrum and bin are valid")[0];
                if reference.abs() < 0.5 {
                    accepted += 1;
                    check_real_estimate(
                        actual.expect("image shifts the estimate inside"),
                        &tone,
                        len,
                        bin,
                        "exact half-bin",
                    );
                } else {
                    rejected += 1;
                    assert_eq!(actual, None, "image shifts the estimate outside");
                }
            }
        }
    }
    assert!(
        accepted > 0 && rejected > 0,
        "exercise both image-bias directions"
    );
}

#[test]
fn exact_half_bin_classification_tracks_the_image_for_every_scalar() {
    exact_half_bin_images::<f64>();
    exact_half_bin_images::<f32>();
}

fn sparse_on_bin_cosine<T: RealField>(len: usize, bin: usize) {
    let n = T::from_f64(len as f64);
    let mut spectrum = vec![Complex::new(T::ZERO, T::ZERO); len];
    let coefficient = Complex::new(n / (T::ONE + T::ONE), T::ZERO);
    spectrum[bin] = coefficient;
    spectrum[len - bin] = coefficient;

    let estimate = estimate_peaks(&spectrum, &[bin], NonZeroUsize::MIN)
        .expect("invariant: the sparse spectrum and its peak bin are valid")[0]
        .unwrap_or_else(|| panic!("on-bin cosine at {bin}/{len} was rejected"));
    // Exact: the zero neighbours give a zero offset, the image kernel at a
    // non-zero integer argument is zero, and `R(0) = N`, so the solve returns
    // the bin's value over `N` without rounding.
    assert_eq!(estimate.position().to_f64(), bin as f64, "on-bin position");
    assert_eq!(estimate.amplitude().to_f64(), 1.0, "on-bin amplitude");
    assert_eq!(estimate.phase().to_f64(), 0.0, "on-bin phase");
}

#[test]
fn sparse_on_bin_cosines_remain_solvable_at_large_and_odd_endpoints() {
    // At this valid f32 length, `4 ε (2k + 3) > 1` for the upper image bin.
    // A size-based determinant heuristic therefore rejects a nonsingular
    // system even though the direct kernel is N and the image kernel is zero.
    let len = 1 << 21;
    sparse_on_bin_cosine::<f32>(len, 3 * len / 4);

    sparse_on_bin_cosine::<f64>(128, 96);
    // Odd frames have no Nyquist bin. One bin farther from the midpoint,
    // the image is outside the three-bin read and this exact solve applies.
    sparse_on_bin_cosine::<f64>(127, 62);
    sparse_on_bin_cosine::<f32>(127, 62);
}

#[test]
fn adjacent_images_at_the_odd_midpoint_are_unresolved() {
    // At N=127 a cosine at 64 has X[63]=X[64]=N/2. The three-bin
    // ratios at 63 and 64 are respectively -1 and +1: the closure gives
    // offsets -1 and +1, outside the half-bin. This is image interference,
    // not a Nyquist singularity, and neither bin is an isolated peak.
    let mut spectrum = vec![Complex64::new(0.0, 0.0); 127];
    spectrum[63] = Complex64::new(63.5, 0.0);
    spectrum[64] = Complex64::new(63.5, 0.0);
    for bin in [63, 64] {
        assert_eq!(
            estimate_peaks(&spectrum, &[bin], NonZeroUsize::MIN)
                .expect("invariant: the midpoint bin is in range"),
            vec![None]
        );
    }
}

/// Near Nyquist in a long frame the image's argument is about `N`, where f32
/// holds only a quarter bin at `N = 2^21`. Formed from the signed residue of
/// `2k` modulo `N`, it keeps the offset's precision on both sides of Nyquist,
/// so the f32 estimate follows the f64 one. The three bins, evaluated in f64
/// and narrowed, differ by `ε₃₂` relative, which moves the ratio by at most
/// `2ε₃₂` (the ratio bound above); the ratio and inverse-tangent arithmetic
/// add `8ε₃₂`, so the offset moves by at most `10ε₃₂` bins and the amplitude
/// by `(2 + π)` times that through the kernel slopes; the solve's arithmetic
/// adds `16ε₃₂` and the peak bin's own rounding `E / (|R| − |I|)`, under
/// `2ε₃₂` here: `(2 + π) · 10 + 16 + 2 < 72` times `ε₃₂` of the amplitude.
fn long_frame_image_keeps_the_offset_precision_in_f32(offset: f64, bin_from_half: isize) {
    let len: usize = 1 << 21;
    let tone = Tone {
        position: (len / 2) as f64 + offset,
        amplitude: 1.0,
        phase: 0.4,
    };
    let bin = (len / 2)
        .checked_add_signed(bin_from_half)
        .expect("invariant: near Nyquist");
    // The two scalars read the same three bins, so they come from the model's
    // closed-form kernel (checked against the defining sum by `tones`) rather
    // than from 2^21-term sums.
    let model = Model::new(len);
    let bins: Vec<(usize, Complex64)> = (bin - 1..=bin + 1)
        .map(|p| (p, model.contribution(tone.position, tone.a(), p as f64)))
        .collect();
    let mut wide = vec![Complex64::new(0.0, 0.0); len];
    let mut narrow = vec![Complex::new(0.0_f32, 0.0); len];
    for &(p, value) in &bins {
        wide[p] = value;
        narrow[p] = Complex::new(
            <f32 as eunomia::FloatElement>::from_f64(value.re),
            <f32 as eunomia::FloatElement>::from_f64(value.im),
        );
    }
    let wide = estimate_peaks(&wide, &[bin], NonZeroUsize::MIN)
        .expect("invariant: the long frame and its bin are valid")[0]
        .expect("the f64 reading resolves the tone");
    let narrow = estimate_peaks(&narrow, &[bin], NonZeroUsize::MIN)
        .expect("invariant: the long frame and its bin are valid")[0]
        .expect("the f32 reading resolves the tone");
    let difference = (f64::from(narrow.amplitude()) - wide.amplitude()).abs();
    let bound = 72.0 * f64::from(f32::EPSILON) * wide.amplitude();
    assert!(
        difference <= bound,
        "bin {bin}: f32 amplitude {} departs from f64 {} by {difference:e} > {bound:e}",
        narrow.amplitude(),
        wide.amplitude()
    );
}

#[test]
fn long_frame_images_keep_the_offset_precision_in_f32_on_both_sides_of_nyquist() {
    long_frame_image_keeps_the_offset_precision_in_f32(0.9, 1);
    long_frame_image_keeps_the_offset_precision_in_f32(-0.7, -1);
    long_frame_image_keeps_the_offset_precision_in_f32(-2.3, -2);
}
