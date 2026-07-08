//! Pure metric and timing helpers used across the validation suite.
//!
//! Each helper is a value-semantic comparison or timing function with no
//! external state — extracted for SRP isolation so the orchestrating suite
//! module remains focused on report assembly.

use eunomia::Complex64;
use leto::{Array1, Array3};
use std::time::Instant;

/// Synthesised deterministic 1D real signal used by orchestrator paths that
/// need a representative real input.
pub(super) fn representative_signal_1d(len: usize) -> Array1<f64> {
    Array1::from(
        (0..len)
            .map(|i| {
                let x = i as f64;
                (0.31 * x).sin() + 0.17 * (0.73 * x).cos()
            })
            .collect::<Vec<_>>(),
    )
}

/// Synthesised deterministic real field used as a reference signal for
/// precision-profile comparisons.
pub(super) fn representative_field_3d(shape: [usize; 3]) -> Array3<f64> {
    Array3::from_shape_fn(shape, |[i, j, k]| {
        let x = i as f64;
        let y = j as f64;
        let z = k as f64;
        (0.11 * x).sin() + (0.13 * y).cos() + (0.17 * z).sin()
    })
}

/// Maximum elementwise complex-norm difference between two sequences.
pub(super) fn max_complex_abs_delta<'a, I, J>(left: I, right: J) -> f64
where
    I: IntoIterator<Item = &'a Complex64>,
    J: IntoIterator<Item = &'a Complex64>,
{
    left.into_iter()
        .zip(right)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(0.0, f64::max)
}

/// Maximum elementwise absolute difference between two 1D real arrays.
pub(super) fn max_real_abs_delta(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

/// Maximum elementwise absolute difference between two 3D real arrays.
pub(super) fn max_real_abs_delta_3d(left: &Array3<f64>, right: &Array3<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

/// Maximum elementwise relative complex error, with denominator floored at 1.0
/// to avoid blow-up near zero-magnitude reference values.
pub(super) fn relative_complex_error<'a, I, J>(left: I, right: J) -> f64
where
    I: IntoIterator<Item = &'a Complex64>,
    J: IntoIterator<Item = &'a Complex64>,
{
    left.into_iter()
        .zip(right)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm() / lhs.norm().max(1.0))
        .fold(0.0, f64::max)
}

/// Time a closure in milliseconds. The closure's result is discarded.
pub(super) fn elapsed_ms<F, T>(f: F) -> f64
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let _ = f();
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::Complex64;

    #[test]
    fn max_complex_abs_delta_returns_zero_for_identical_streams() {
        let a = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
        assert_eq!(max_complex_abs_delta(a.iter(), a.iter()), 0.0);
    }

    #[test]
    fn max_complex_abs_delta_returns_largest_norm_difference() {
        let a = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let b = [Complex64::new(1.0, 0.0), Complex64::new(3.0, 4.0)];
        assert_eq!(max_complex_abs_delta(a.iter(), b.iter()), 5.0);
    }

    #[test]
    fn max_real_abs_delta_handles_signs_and_zero() {
        let a = Array1::from(vec![1.0, -2.0, 3.0]);
        let b = Array1::from(vec![1.0, 2.0, 3.0]);
        assert_eq!(max_real_abs_delta(&a, &b), 4.0);
    }

    #[test]
    fn relative_complex_error_floors_denominator_at_one() {
        let lhs = [Complex64::new(0.1, 0.0)];
        let rhs = [Complex64::new(0.6, 0.0)];
        let err = relative_complex_error(lhs.iter(), rhs.iter());
        assert!((err - 0.5).abs() < 1e-15, "expected 0.5, got {err}");
    }

    #[test]
    fn representative_field_3d_is_deterministic_and_shape_preserving() {
        let a = representative_field_3d([2, 3, 4]);
        let b = representative_field_3d([2, 3, 4]);
        assert_eq!(a.shape(), [2, 3, 4]);
        assert_eq!(a, b);
    }
}
