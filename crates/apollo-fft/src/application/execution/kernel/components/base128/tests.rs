//! Correctness for the 128-point base butterfly. The direct DFT is the
//! analytical authority.

use super::butterfly::transform_128;
use eunomia::Complex64;
use std::f64::consts::TAU;

fn dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (t, v) in input.iter().enumerate() {
                let (s, c) = (sign * TAU * ((k * t) % n) as f64 / n as f64).sin_cos();
                re += v.re * c - v.im * s;
                im += v.re * s + v.im * c;
            }
            Complex64::new(re, im)
        })
        .collect()
}

fn signal() -> Vec<Complex64> {
    (0..128)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.043 * x).sin(), 0.25 * (0.029 * x).cos())
        })
        .collect()
}

fn tolerance(input: &[Complex64]) -> f64 {
    let l1: f64 = input.iter().map(|v| v.re.hypot(v.im)).sum();
    16.0 * 7.0 * (f64::EPSILON / 2.0) * l1
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
        transform_128::<f64, false>(&mut data),
        "this host's dispatched width must run the base butterfly"
    );
    let (err, bound) = (worst(&data, &dft(&src, false)), tolerance(&src));
    assert!(err <= bound, "forward differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn inverse_matches_the_direct_transform() {
    let src = signal();
    let mut data = src.clone();
    assert!(transform_128::<f64, true>(&mut data));
    let (err, bound) = (worst(&data, &dft(&src, true)), tolerance(&src));
    assert!(err <= bound, "inverse differs by {err:.3e} > {bound:.3e}");
}

#[test]
fn forward_then_inverse_recovers_the_input() {
    let src = signal();
    let mut data = src.clone();
    assert!(transform_128::<f64, false>(&mut data));
    assert!(transform_128::<f64, true>(&mut data));
    let n = 128.0;
    let bound = tolerance(&src) * n;
    let err = data
        .iter()
        .zip(src.iter())
        .map(|(a, b)| (a.re - b.re * n).hypot(a.im - b.im * n))
        .fold(0.0f64, f64::max);
    assert!(
        err <= bound,
        "round trip differs by {err:.3e} > {bound:.3e}"
    );
}

#[test]
fn matches_the_production_route_within_rounding() {
    let src = signal();
    let mut ours = src.clone();
    assert!(transform_128::<f64, false>(&mut ours));

    let mut theirs = src.clone();
    let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D { n: 128 });
    plan.forward_complex_slice_inplace(&mut theirs);

    let bound = 2.0 * tolerance(&src);
    let err = worst(&ours, &theirs);
    assert!(err <= bound, "routes differ by {err:.3e} > {bound:.3e}");
}
