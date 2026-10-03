//! Inputs, closed-form references and component-wise assertions.

use super::bound::{impulse_bound, roundoff, Pass};
use core::f64::consts::TAU;
use eunomia::{Complex, FloatElement};

pub(super) fn assert_close<T>(actual: &[Complex<T>], expected: &[(f64, f64)], tolerance: f64)
where
    T: FloatElement,
{
    assert_eq!(actual.len(), expected.len());
    for (index, (value, &(re, im))) in actual.iter().zip(expected).enumerate() {
        let got = (value.re.to_f64(), value.im.to_f64());
        assert!(
            (got.0 - re).abs() <= tolerance && (got.1 - im).abs() <= tolerance,
            "bin {index}: {got:?} differs from ({re}, {im}) by more than {tolerance}"
        );
    }
}

/// Asserts every component of an impulse response against `reference / N`
/// (`reference` for the forward pass) under its own `impulse_bound`.
pub(super) fn assert_impulse_response<T: FloatElement>(
    actual: &[Complex<T>],
    reference: &[(f64, f64)],
    pass: Pass,
) {
    let n = actual.len();
    assert_eq!(n, reference.len());
    let divisor = match pass {
        Pass::Forward => 1.0,
        Pass::Inverse => f64::from_count(n),
    };
    for (bin, (value, &(re, im))) in actual.iter().zip(reference).enumerate() {
        for (got, unit) in [(value.re.to_f64(), re), (value.im.to_f64(), im)] {
            let bound = impulse_bound(roundoff::<T>(), unit, pass, n);
            let want = unit / divisor;
            assert!(
                (got - want).abs() <= bound,
                "length {n} bin {bin}: {got:e} differs from {want:e} by more than {bound:e}"
            );
        }
    }
}

pub(super) fn signal<T: FloatElement>(values: &[(f64, f64)]) -> Vec<Complex<T>> {
    values
        .iter()
        .map(|&(re, im)| Complex::new(T::from_f64(re), T::from_f64(im)))
        .collect()
}

/// `n` zeros with `(1, 0)` at `index`.
pub(super) fn impulse<T: FloatElement>(n: usize, index: usize) -> Vec<Complex<T>> {
    let mut values = vec![(0.0, 0.0); n];
    values[index] = (1.0, 0.0);
    signal::<T>(&values)
}

/// `exp(sign · 2π i k / N)` for every bin `k`, generated through
/// eunomia's libm-backed sine and cosine exactly as the kernel does.
pub(super) fn unit_circle(n: usize, sign: f64) -> Vec<(f64, f64)> {
    (0..n)
        .map(|k| {
            let angle = sign * TAU * f64::from_count(k) / f64::from_count(n);
            (FloatElement::cos(angle), FloatElement::sin(angle))
        })
        .collect()
}
