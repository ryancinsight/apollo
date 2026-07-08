//! CZT metadata: validated starting point, frequency step, and transform parameters.
//!
//! The chirp z-transform evaluates the z-transform at M points on a logarithmic spiral:
//! `z_k = A * W^{-k}`, `k = 0, 1, ..., M-1`,
//! where `A` is the starting point and `W` is the frequency step ratio.

use crate::domain::contracts::error::CztError;
use eunomia::Complex64;
use serde::{Deserialize, Serialize};

/// Validated CZT parameters: input length, output length, and spiral parameters.
///
/// The z-transform is evaluated at z_k = A * W^{-k} for k = 0..M.
/// Setting A = 1, W = exp(-2*pi*i/N), M = N recovers the standard DFT.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CztParameters {
    input_len: usize,
    output_len: usize,
    a: Complex64,
    w: Complex64,
}

impl CztParameters {
    /// Create validated CZT parameters.
    ///
    /// Returns CztError::EmptyLength if either length is zero.
    /// Returns CztError::InvalidParameters if either spiral parameter has zero
    /// magnitude or if any component of `a` or `w` is non-finite.
    pub fn new(
        input_len: usize,
        output_len: usize,
        a: Complex64,
        w: Complex64,
    ) -> Result<Self, CztError> {
        if input_len == 0 || output_len == 0 {
            return Err(CztError::EmptyLength);
        }
        let a_norm = a.norm();
        let w_norm = w.norm();
        if a_norm == 0.0 || w_norm == 0.0 || !a_norm.is_finite() || !w_norm.is_finite() {
            return Err(CztError::InvalidParameters);
        }
        Ok(Self {
            input_len,
            output_len,
            a,
            w,
        })
    }

    /// Input signal length.
    #[must_use]
    pub const fn input_len(&self) -> usize {
        self.input_len
    }

    /// Output (frequency samples) length.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.output_len
    }

    /// Starting point on the z-plane spiral.
    #[must_use]
    pub const fn a(&self) -> Complex64 {
        self.a
    }

    /// Frequency step ratio.
    #[must_use]
    pub const fn w(&self) -> Complex64 {
        self.w
    }

    /// Return DFT-equivalent parameters: A=1, W=exp(-2*pi*i/N), M=N.
    ///
    /// # Theorem
    /// With A=1 and W=exp(-2*pi*i/N), the CZT evaluates the z-transform at
    /// equally-spaced unit-circle points, which is exactly the N-point DFT.
    pub fn dft_parameters(n: usize) -> Result<Self, CztError> {
        if n == 0 {
            return Err(CztError::EmptyLength);
        }
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        Self::new(n, n, Complex64::new(1.0, 0.0), w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_input_len() {
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::new(0.9, 0.1);
        assert!(matches!(
            CztParameters::new(0, 4, a, w),
            Err(CztError::EmptyLength)
        ));
    }

    #[test]
    fn new_rejects_zero_output_len() {
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::new(0.9, 0.1);
        assert!(matches!(
            CztParameters::new(4, 0, a, w),
            Err(CztError::EmptyLength)
        ));
    }

    #[test]
    fn new_rejects_zero_magnitude_a() {
        let a = Complex64::new(0.0, 0.0);
        let w = Complex64::new(0.9, 0.1);
        assert!(matches!(
            CztParameters::new(4, 4, a, w),
            Err(CztError::InvalidParameters)
        ));
    }

    #[test]
    fn new_rejects_zero_magnitude_w() {
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::new(0.0, 0.0);
        assert!(matches!(
            CztParameters::new(4, 4, a, w),
            Err(CztError::InvalidParameters)
        ));
    }

    #[test]
    fn dft_parameters_roundtrip() {
        let p = CztParameters::dft_parameters(8).unwrap();
        assert_eq!(p.input_len(), 8);
        assert_eq!(p.output_len(), 8);
        let expected_w = Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0);
        assert!((p.w() - expected_w).norm() < 1e-14);
        assert!((p.a() - Complex64::new(1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn dft_parameters_rejects_zero_n() {
        assert!(matches!(
            CztParameters::dft_parameters(0),
            Err(CztError::EmptyLength)
        ));
    }
}
