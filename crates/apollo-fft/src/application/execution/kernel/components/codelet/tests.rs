//! Correctness for the N = 16 register-resident codelet.
//!
//! The primary oracle is a compensated direct DFT — analytical, independent of
//! every other route in the crate. The differential against the incumbent
//! sized path is secondary: it pins that a dispatch swap cannot change results
//! beyond rounding, with the bound derived from the stage counts rather than
//! observed.

use super::try_transform_16;
use eunomia::Complex64;
use std::f64::consts::TAU;

fn signal() -> Vec<Complex64> {
    (0..16)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.4 * x + 0.3).sin(), (0.7 * x - 0.1).cos())
        })
        .collect()
}

/// Direct DFT accumulated in f64; at N = 16 its `O(N * u)` error is far below
/// the assertion bound.
fn dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (t, v) in input.iter().enumerate() {
                let (s, c) = (sign * TAU * ((k * t) % n) as f64 / n as f64).sin_cos();
                re += v.re * c - v.im * s;
                im += v.re * s + v.im * c;
            }
            Complex64::new(re, im)
        })
        .collect()
}

/// `O(stages * u * l1)` with the shared factor the crate's other accuracy
/// tests use.
fn bound(input: &[Complex64]) -> f64 {
    let l1: f64 = input.iter().map(|v| v.re.hypot(v.im)).sum();
    16.0 * 4.0 * (f64::EPSILON / 2.0) * l1
}

fn worst(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.re - y.re).hypot(x.im - y.im))
        .fold(0.0f64, f64::max)
}

#[test]
fn forward_matches_the_direct_transform() {
    let src = signal();
    let mut data = src.clone();
    assert!(
        try_transform_16::<f64, false, false>(&mut data),
        "this host's dispatched width must run the codelet"
    );
    let expected = dft(&src, false);
    let err = worst(&data, &expected);
    let limit = bound(&src);
    assert!(err <= limit, "forward differs by {err:.3e} > {limit:.3e}");
}

#[test]
fn inverse_matches_the_direct_transform() {
    let src = signal();
    let mut data = src.clone();
    assert!(try_transform_16::<f64, true, false>(&mut data));
    let expected = dft(&src, true);
    let err = worst(&data, &expected);
    let limit = bound(&src);
    assert!(err <= limit, "inverse differs by {err:.3e} > {limit:.3e}");
}

#[test]
fn normalized_round_trip_recovers_the_input() {
    let src = signal();
    let mut data = src.clone();
    assert!(try_transform_16::<f64, false, false>(&mut data));
    assert!(try_transform_16::<f64, true, true>(&mut data));
    let err = worst(&data, &src);
    let limit = 2.0 * bound(&src);
    assert!(
        err <= limit,
        "round trip differs by {err:.3e} > {limit:.3e}"
    );
}

#[test]
fn matches_the_incumbent_route_within_rounding() {
    let src = signal();
    let mut ours = src.clone();
    assert!(try_transform_16::<f64, false, false>(&mut ours));

    let mut theirs = src.clone();
    crate::FftPlan1D::<f64>::new(crate::Shape1D { n: 16 })
        .forward_complex_slice_inplace(&mut theirs);

    let err = worst(&ours, &theirs);
    // Two different evaluation orders, each within the stage bound.
    let limit = 2.0 * bound(&src);
    assert!(
        err <= limit,
        "codelet differs from the route by {err:.3e} > {limit:.3e}"
    );
}
