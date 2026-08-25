//! Differential verification of Apollo's power-of-two forward FFT against PhastFT.
//!
//! ## Why an external engine and not only the direct DFT
//!
//! Apollo already checks its transforms against an O(N^2) direct DFT, which is
//! the authoritative analytical oracle but is too slow to reach the sizes where
//! a staging, blocking, or bit-reversal defect first appears. PhastFT is an
//! independently authored power-of-two engine — radix-2 decimation-in-time with
//! CO-BRAVO bit reversal over split real/imaginary planes, against Apollo's
//! self-sorting Stockham over interleaved samples. The two share no code and no
//! data layout, so agreement at 2^12 is evidence about the parts a small-N
//! analytical case cannot reach.
//!
//! ## Convention pinning
//!
//! Both engines are exercised with the same signal and the forward direction.
//! The transform sign, normalization, and output ordering conventions must agree
//! for the comparison to be meaningful; that agreement is what these assertions
//! pin. A convention difference is a real finding, not a tolerance to widen.
//!
//! ## Tolerance derivation
//!
//! For a fast Fourier transform of length `N = 2^m` evaluated in binary floating
//! point with unit roundoff `u`, the componentwise forward error is bounded by
//! `c * m * u * ||x||_1` for a modest constant `c` — the `O(log N * u)` growth of
//! Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section
//! 24.1, with `|y_k| <= ||x||_1` bounding each output bin. Two independent
//! implementations each lie within that ball, so their difference is bounded by
//! twice it. [`TOLERANCE_FACTOR`] carries `2c`; it is a bound on the difference
//! of two correct implementations, not a knob fitted to observed error, and the
//! reported margin below shows how much of it is actually consumed.
//!
//! The two engines fix different reduction orders — Stockham stages against
//! radix-2 DIT stages on bit-reversed input — so bitwise equality is not a valid
//! oracle here and is not asserted.

use eunomia::{Complex32, Complex64};
use phastft::planner::{Direction, PlannerDit32, PlannerDit64};
use phastft::{fft_f32_dit_with_planner, fft_f64_dit_with_planner};

/// `2c` from the derivation above. Covers both engines' error balls plus the
/// difference in how each generates its twiddle factors.
const TOLERANCE_FACTOR: f64 = 16.0;

/// Powers of two from the smallest non-trivial transform through a length whose
/// working set leaves L1, so blocking and staging paths are exercised rather
/// than only the codelets.
const SIZES: [usize; 7] = [4, 8, 16, 64, 256, 1_024, 4_096];

fn signal(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

/// `||x||_1` over complex magnitudes, the bound on any single output bin.
fn l1_norm(input: &[Complex64]) -> f64 {
    input.iter().map(|v| v.re.hypot(v.im)).sum()
}

/// `c * m * u * ||x||_1` doubled, per the module derivation.
fn tolerance(len: usize, l1: f64, unit_roundoff: f64) -> f64 {
    let stages = f64::from(u32::try_from(len.trailing_zeros()).expect("power of two fits u32"));
    TOLERANCE_FACTOR * stages * unit_roundoff * l1
}

#[test]
fn forward_f64_agrees_with_phastft_within_derived_bound() {
    // f64 unit roundoff: 2^-53.
    const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

    for len in SIZES {
        let source = signal(len);
        let l1 = l1_norm(&source);

        let mut apollo = source.clone();
        apollo_fft::FftPlan1D::<f64>::new(apollo_fft::Shape1D { n: len })
            .forward_complex_slice_inplace(&mut apollo);

        let mut re: Vec<f64> = source.iter().map(|v| v.re).collect();
        let mut im: Vec<f64> = source.iter().map(|v| v.im).collect();
        fft_f64_dit_with_planner(
            &mut re,
            &mut im,
            Direction::Forward,
            &PlannerDit64::new(len),
        );

        let bound = tolerance(len, l1, UNIT_ROUNDOFF);
        let (worst, bin) = apollo
            .iter()
            .zip(re.iter().zip(im.iter()))
            .enumerate()
            .map(|(k, (a, (pr, pi)))| ((a.re - pr).hypot(a.im - pi), k))
            .fold(
                (0.0_f64, 0_usize),
                |acc, item| {
                    if item.0 > acc.0 {
                        item
                    } else {
                        acc
                    }
                },
            );

        assert!(
            worst <= bound,
            "N={len}: Apollo and PhastFT forward f64 differ by {worst:.3e} at bin {bin}, \
             exceeding the derived bound {bound:.3e}. Two engines outside a bound both \
             satisfy is a defect in one of them or a convention mismatch between them; \
             it is not a tolerance to widen."
        );
        // A margin at or near 1.0 would mean the bound is being fitted rather
        // than derived, so it is reported for the reviewer rather than hidden.
        eprintln!(
            "N={len}: worst |Apollo - PhastFT| = {worst:.3e}, bound {bound:.3e}, \
             margin consumed {:.4}",
            worst / bound
        );
    }
}

#[test]
fn forward_f32_agrees_with_phastft_within_derived_bound() {
    // f32 unit roundoff: 2^-24, applied to the same derivation.
    const UNIT_ROUNDOFF: f64 = (f32::EPSILON / 2.0) as f64;

    for len in SIZES {
        let source64 = signal(len);
        let source: Vec<Complex32> = source64
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        let l1: f64 = source.iter().map(|v| f64::from(v.re.hypot(v.im))).sum();

        let mut apollo = source.clone();
        apollo_fft::FftPlan1D::<f32>::new(apollo_fft::Shape1D { n: len })
            .forward_complex_slice_inplace(&mut apollo);

        let mut re: Vec<f32> = source.iter().map(|v| v.re).collect();
        let mut im: Vec<f32> = source.iter().map(|v| v.im).collect();
        fft_f32_dit_with_planner(
            &mut re,
            &mut im,
            Direction::Forward,
            &PlannerDit32::new(len),
        );

        let bound = tolerance(len, l1, UNIT_ROUNDOFF);
        let worst = apollo
            .iter()
            .zip(re.iter().zip(im.iter()))
            .map(|(a, (pr, pi))| f64::from((a.re - pr).hypot(a.im - pi)))
            .fold(0.0_f64, f64::max);

        assert!(
            worst <= bound,
            "N={len}: Apollo and PhastFT forward f32 differ by {worst:.3e}, exceeding the \
             derived bound {bound:.3e}."
        );
        eprintln!(
            "N={len}: worst |Apollo - PhastFT| f32 = {worst:.3e}, bound {bound:.3e}, \
             margin consumed {:.4}",
            worst / bound
        );
    }
}
