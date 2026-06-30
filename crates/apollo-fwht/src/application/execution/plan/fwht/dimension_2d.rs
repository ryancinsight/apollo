//! 2D Fast Walsh-Hadamard Transform plan.
//!
//! The 2D WHT is separable: apply the 1D WHT to each row, then to each column.
//!
//! ## Mathematical contract
//!
//! The N×N unnormalized 2D WHT satisfies
//! `W_{2D}[m,n] = Σ_j Σ_k x[j,k] (-1)^{popcount(j&m)+popcount(k&n)}`.
//!
//! Since WHT is involutory (WHT(WHT(x)) = N·x), the 2D WHT applied twice gives
//! `W_{2D}(W_{2D}(X)) = N²·X`, and the inverse is `(1/N²)·forward_2d`.
//!
//! Both N (rows) and N (cols) must be equal and a power of two.

use crate::application::execution::kernel::direct::wht_inplace;
use crate::domain::contracts::error::FwhtError;
use leto::Array2;
use eunomia::Complex64;
use serde::{Deserialize, Serialize};

/// Reusable 2D FWHT plan.
///
/// Both spatial dimensions must be equal to `n`, which must be a power of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FwhtPlan2D {
    n: usize,
}

impl FwhtPlan2D {
    /// Create a validated 2D FWHT plan for an `n×n` grid.
    ///
    /// # Errors
    /// Returns [`FwhtError::EmptyInput`] if `n == 0`.
    /// Returns [`FwhtError::NonPowerOfTwo`] if `n` is not a power of two.
    pub fn new(n: usize) -> Result<Self, FwhtError> {
        if n == 0 {
            return Err(FwhtError::EmptyInput);
        }
        if !n.is_power_of_two() {
            return Err(FwhtError::NonPowerOfTwo);
        }
        Ok(Self { n })
    }

    /// Return the side length of the `n×n` grid.
    #[must_use]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Return whether the plan length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Forward 2D WHT of a real-valued `n×n` array. O(N² log N).
    ///
    /// Applies the 1D WHT to each row, then to each column.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `input.shape() != (n, n)`.
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>, FwhtError> {
        let n = self.n;
        let [rows, cols] = input.shape();
        if rows != n || cols != n {
            return Err(FwhtError::LengthMismatch);
        }
        // Row pass: apply 1D WHT to each row.
        let mut tmp = input.clone();
        for r in 0..n {
            let mut row: Vec<f64> = (0..n).map(|c| tmp[[r, c]]).collect();
            wht_inplace(&mut row);
            for c in 0..n {
                tmp[[r, c]] = row[c];
            }
        }
        // Column pass: apply 1D WHT to each column.
        for c in 0..n {
            let mut col: Vec<f64> = (0..n).map(|r| tmp[[r, c]]).collect();
            wht_inplace(&mut col);
            for r in 0..n {
                tmp[[r, c]] = col[r];
            }
        }
        Ok(tmp)
    }

    /// Forward 2D WHT into a caller-owned output buffer.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when dimensions do not match the plan.
    pub fn forward_into(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
    ) -> Result<(), FwhtError> {
        let result = self.forward(input)?;
        output.assign(&result.view());
        Ok(())
    }

    /// Forward 2D WHT in-place. O(N² log N) butterfly operations.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `data.shape() != (n, n)`.
    pub fn forward_inplace(&self, data: &mut Array2<f64>) -> Result<(), FwhtError> {
        let n = self.n;
        let [rows, cols] = data.shape();
        if rows != n || cols != n {
            return Err(FwhtError::LengthMismatch);
        }
        // Row pass.
        for r in 0..n {
            let mut row: Vec<f64> = (0..n).map(|c| data[[r, c]]).collect();
            wht_inplace(&mut row);
            for c in 0..n {
                data[[r, c]] = row[c];
            }
        }
        // Column pass.
        for c in 0..n {
            let mut col: Vec<f64> = (0..n).map(|r| data[[r, c]]).collect();
            wht_inplace(&mut col);
            for r in 0..n {
                data[[r, c]] = col[r];
            }
        }
        Ok(())
    }

    /// Inverse 2D WHT: applies forward WHT then divides by N².
    ///
    /// Correctness: `WHT_2D(WHT_2D(X)) = N²·X`, so scaling by `1/N²` recovers X.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `input.shape() != (n, n)`.
    pub fn inverse(&self, input: &Array2<f64>) -> Result<Array2<f64>, FwhtError> {
        let n = self.n;
        let mut result = self.forward(input)?;
        let scale = 1.0 / (n * n) as f64;
        result.mapv_inplace(|v| v * scale);
        Ok(result)
    }

    /// Inverse 2D WHT into a caller-owned output buffer.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when dimensions do not match the plan.
    pub fn inverse_into(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
    ) -> Result<(), FwhtError> {
        let result = self.inverse(input)?;
        output.assign(&result.view());
        Ok(())
    }

    /// Inverse 2D WHT in-place.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `data.shape() != (n, n)`.
    pub fn inverse_inplace(&self, data: &mut Array2<f64>) -> Result<(), FwhtError> {
        let n = self.n;
        self.forward_inplace(data)?;
        let scale = 1.0 / (n * n) as f64;
        data.mapv_inplace(|v| v * scale);
        Ok(())
    }

    /// Forward 2D WHT of a complex-valued `n×n` array.
    ///
    /// Applies the real-valued WHT butterfly independently to the real and imaginary
    /// parts, which preserves correctness because WHT is linear over ℝ and ℂ.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `input.shape() != (n, n)`.
    pub fn forward_complex(
        &self,
        input: &Array2<Complex64>,
    ) -> Result<Array2<Complex64>, FwhtError> {
        let n = self.n;
        let [rows, cols] = input.shape();
        if rows != n || cols != n {
            return Err(FwhtError::LengthMismatch);
        }
        let re_in: Array2<f64> = input.mapv(|v| v.re);
        let im_in: Array2<f64> = input.mapv(|v| v.im);
        let re_out = self.forward(&re_in)?;
        let im_out = self.forward(&im_in)?;
        Ok(re_out.zip_map(&im_out, |re, im| Complex64::new(re, im)))
    }

    /// Inverse 2D WHT of a complex-valued `n×n` spectrum.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `input.shape() != (n, n)`.
    pub fn inverse_complex(
        &self,
        input: &Array2<Complex64>,
    ) -> Result<Array2<Complex64>, FwhtError> {
        let n = self.n;
        let [rows, cols] = input.shape();
        if rows != n || cols != n {
            return Err(FwhtError::LengthMismatch);
        }
        let re_in: Array2<f64> = input.mapv(|v| v.re);
        let im_in: Array2<f64> = input.mapv(|v| v.im);
        let re_out = self.inverse(&re_in)?;
        let im_out = self.inverse(&im_in)?;
        Ok(re_out.zip_map(&im_out, |re, im| Complex64::new(re, im)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn plan_new_rejects_zero_and_non_power_of_two() {
        assert_eq!(FwhtPlan2D::new(0).unwrap_err(), FwhtError::EmptyInput);
        assert_eq!(FwhtPlan2D::new(3).unwrap_err(), FwhtError::NonPowerOfTwo);
        assert!(FwhtPlan2D::new(4).is_ok());
    }

    /// WHT_2D(WHT_2D(X)) = N²·X (involution).
    #[test]
    fn forward_2d_involution_equals_n_squared_times_input() {
        let n = 4_usize;
        let plan = FwhtPlan2D::new(n).expect("plan");
        // Analytically constructed: use a non-trivial non-zero signal.
        let input = leto::Array2::from_shape_vec([4, 4], vec![1.0_f64, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0, 0.1, -0.3, 1.2, -0.5, 2.1, -1.1, 0.7]).unwrap();
        let first = plan.forward(&input).expect("first forward");
        let second = plan.forward(&first).expect("second forward");
        let n2 = (n * n) as f64;
        for r in 0..n {
            for c in 0..n {
                assert_abs_diff_eq!(second[[r, c]], n2 * input[[r, c]], epsilon = 1.0e-10);
            }
        }
    }

    /// inverse(forward(X)) = X for all real inputs.
    #[test]
    fn inverse_2d_roundtrip_recovers_signal() {
        let n = 4_usize;
        let plan = FwhtPlan2D::new(n).expect("plan");
        let input = leto::Array2::from_shape_vec([4, 4], vec![1.0_f64, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0, 0.1, -0.3, 1.2, -0.5, 2.1, -1.1, 0.7]).unwrap();
        let spectrum = plan.forward(&input).expect("forward");
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for r in 0..n {
            for c in 0..n {
                assert_abs_diff_eq!(recovered[[r, c]], input[[r, c]], epsilon = 1.0e-12);
            }
        }
    }

    /// 2D WHT of a separable signal: outer product x⊗y.
    /// W_{2D}(x⊗y) = WHT(x) ⊗ WHT(y) by separability.
    #[test]
    fn forward_2d_matches_separable_outer_product() {
        use super::super::dimension_1d::FwhtPlan;
        use leto::Array1;
        let n = 4_usize;
        let plan_2d = FwhtPlan2D::new(n).expect("2D plan");
        let plan_1d = FwhtPlan::new(n).expect("1D plan");
        let row = Array1::from(vec![1.0_f64, -2.0, 0.5, 0.25]);
        let col = Array1::from(vec![3.0_f64, -1.5, -0.75, 0.5]);
        // Build outer product input.
        let mut input = Array2::<f64>::zeros([n, n]);
        for r in 0..n {
            for c in 0..n {
                input[[r, c]] = row[r] * col[c];
            }
        }
        let result = plan_2d.forward(&input).expect("2D forward");
        let row_wht = plan_1d.forward(&row).expect("1D row WHT");
        let col_wht = plan_1d.forward(&col).expect("1D col WHT");
        // Expected: outer product of 1D WHT results.
        for r in 0..n {
            for c in 0..n {
                let expected = row_wht[r] * col_wht[c];
                assert_abs_diff_eq!(result[[r, c]], expected, epsilon = 1.0e-10);
            }
        }
    }

    #[test]
    fn forward_2d_rejects_non_square_input() {
        let plan = FwhtPlan2D::new(4).expect("plan");
        let wrong = Array2::<f64>::zeros([2, 4]);
        assert_eq!(plan.forward(&wrong).unwrap_err(), FwhtError::LengthMismatch);
    }

    #[test]
    fn forward_complex_2d_roundtrip_recovers_signal() {
        let n = 2_usize;
        let plan = FwhtPlan2D::new(n).expect("plan");
        let input = leto::Array2::from_shape_vec([2, 2], vec![Complex64::new(1.0, 0.5), Complex64::new(-1.0, 2.0), Complex64::new(0.25, -0.75), Complex64::new(3.0, -1.5)]).unwrap();
        let spectrum = plan.forward_complex(&input).expect("complex forward");
        let recovered = plan.inverse_complex(&spectrum).expect("complex inverse");
        for r in 0..n {
            for c in 0..n {
                assert_abs_diff_eq!(recovered[[r, c]].re, input[[r, c]].re, epsilon = 1.0e-12);
                assert_abs_diff_eq!(recovered[[r, c]].im, input[[r, c]].im, epsilon = 1.0e-12);
            }
        }
    }
}
