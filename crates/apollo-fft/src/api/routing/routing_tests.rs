//! `supports_length` must agree with the transform it stands in for.
//!
//! The probe checks a tone and Parseval; the evidence that this is enough is a
//! comparison against the direct DFT, which shares no code with either.

use super::supports_length;
use crate::domain::metadata::shape::Shape1D;
use crate::FftPlan1D;
use eunomia::Complex64;

/// Lengths from the tracking item, one per observed failure mode: `361` asserts
/// while planning, `1153` and `6726` assert while executing, and `722`, `1083`,
/// `1444`, `6727`, `6728` return a wrong answer with no diagnostic at all.
const KNOWN_BROKEN: [usize; 8] = [361, 722, 1083, 1153, 1444, 6726, 6727, 6728];

/// Lengths that neighbour the broken ones closely enough that a predicate built
/// on prime-power reasoning would misclassify them: `529 = 23²` and
/// `437 = 19·23` are correct, and both carry the factors that break elsewhere.
const KNOWN_GOOD: [usize; 8] = [380, 437, 529, 760, 1024, 1900, 2048, 4096];

fn direct_dft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .fold(Complex64::new(0.0, 0.0), |acc, (j, &x)| {
                    let angle = -2.0 * std::f64::consts::PI * (k * j % n) as f64 / n as f64;
                    acc + x * Complex64::new(angle.cos(), angle.sin())
                })
        })
        .collect()
}

fn relative_error_against_direct_dft(n: usize) -> Option<f64> {
    let input: Vec<Complex64> = (0..n)
        .map(|i| {
            Complex64::new(
                (i * 37 % 101) as f64 / 101.0 - 0.5,
                (i * 53 % 97) as f64 / 97.0 - 0.5,
            )
        })
        .collect();
    let transformed = std::panic::catch_unwind(|| {
        let mut buffer = input.clone();
        FftPlan1D::<f64>::new(Shape1D::new(n).expect("non-zero length"))
            .forward_complex_slice_inplace(&mut buffer);
        buffer
    })
    .ok()?;
    let reference = direct_dft(&input);
    let error = transformed
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    let scale = reference.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
    Some(error / scale)
}

#[test]
fn broken_lengths_are_rejected() {
    for n in KNOWN_BROKEN {
        assert!(
            !supports_length(n),
            "length {n} is a known-broken route but the probe accepted it"
        );
    }
}

#[test]
fn correct_lengths_are_accepted() {
    for n in KNOWN_GOOD {
        assert!(
            supports_length(n),
            "length {n} transforms correctly but the probe rejected it"
        );
    }
}

#[test]
fn verdicts_match_the_direct_dft() {
    // Direct DFT is O(n²), so the cross-check runs on the lengths where the
    // two families of failure and their correct neighbours all appear.
    for n in KNOWN_BROKEN.into_iter().chain(KNOWN_GOOD) {
        let verdict = supports_length(n);
        match relative_error_against_direct_dft(n) {
            None => assert!(!verdict, "length {n} panics but the probe accepted it"),
            Some(error) => {
                // The correct paths land near 1e-15; the broken ones near 1e0.
                // Any threshold between separates them, so this is a
                // classification, not a tolerance.
                let correct = error < 1e-9;
                assert_eq!(
                    verdict, correct,
                    "length {n}: probe said {verdict}, direct DFT relative error is {error:.3e}"
                );
            }
        }
    }
}

#[test]
fn verdicts_are_cached_and_stable() {
    for n in KNOWN_BROKEN.into_iter().chain(KNOWN_GOOD) {
        assert_eq!(
            supports_length(n),
            supports_length(n),
            "length {n} produced two different verdicts"
        );
    }
}
