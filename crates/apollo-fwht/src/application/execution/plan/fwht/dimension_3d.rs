//! 3D Fast Walsh-Hadamard Transform plan.
//!
//! The 3D WHT is separable: apply the 1D WHT along axis 0, then axis 1, then axis 2.
//!
//! ## Mathematical contract
//!
//! The N×N×N unnormalized 3D WHT satisfies
//! `W_{3D}[l,m,n] = Σ_i Σ_j Σ_k x[i,j,k]·(-1)^{popcount(i&l)+popcount(j&m)+popcount(k&n)}`.
//!
//! Since WHT is involutory (WHT(WHT(x)) = N·x), the 3D WHT applied twice gives N³·X,
//! and the normalized inverse is `(1/N³)·forward_3d`.
//!
//! All three dimensions must equal `n`, which must be a power of two.

use crate::application::execution::kernel::direct::wht_inplace;
use crate::domain::contracts::error::FwhtError;
use eunomia::Complex64;
use leto::Array3;
use serde::{Deserialize, Serialize};

fn scale_array(data: &mut Array3<f64>, scale: f64) {
    for value in data
        .as_slice_mut()
        .expect("invariant: FWHT arrays are contiguous")
    {
        *value *= scale;
    }
}

/// Reusable 3D FWHT plan.
///
/// All spatial dimensions must equal `n`, which must be a power of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FwhtPlan3D {
    n: usize,
}

impl FwhtPlan3D {
    /// Create a validated 3D FWHT plan for an `n×n×n` grid.
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

    /// Return the side length of the `n×n×n` grid.
    #[must_use]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Return whether the plan length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Forward 3D WHT of a real-valued `n×n×n` array. O(N³ log N).
    ///
    /// Applies the 1D WHT along axis 0, then axis 1, then axis 2.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `input.shape() != (n, n, n)`.
    pub fn forward(&self, input: &Array3<f64>) -> Result<Array3<f64>, FwhtError> {
        let n = self.n;
        let [d0, d1, d2] = input.shape();
        if d0 != n || d1 != n || d2 != n {
            return Err(FwhtError::LengthMismatch);
        }
        // Axis-0 pass.
        let mut tmp0 = input.clone();
        for j in 0..n {
            for k in 0..n {
                let mut fiber: Vec<f64> = (0..n).map(|i| tmp0[[i, j, k]]).collect();
                wht_inplace(&mut fiber);
                for i in 0..n {
                    tmp0[[i, j, k]] = fiber[i];
                }
            }
        }
        // Axis-1 pass.
        let mut tmp1 = tmp0;
        for i in 0..n {
            for k in 0..n {
                let mut fiber: Vec<f64> = (0..n).map(|j| tmp1[[i, j, k]]).collect();
                wht_inplace(&mut fiber);
                for j in 0..n {
                    tmp1[[i, j, k]] = fiber[j];
                }
            }
        }
        // Axis-2 pass.
        let mut result = tmp1;
        for i in 0..n {
            for j in 0..n {
                let mut fiber: Vec<f64> = (0..n).map(|k| result[[i, j, k]]).collect();
                wht_inplace(&mut fiber);
                for k in 0..n {
                    result[[i, j, k]] = fiber[k];
                }
            }
        }
        Ok(result)
    }

    /// Forward 3D WHT into a caller-owned output buffer.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when dimensions do not match the plan.
    pub fn forward_into(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
    ) -> Result<(), FwhtError> {
        let result = self.forward(input)?;
        output.assign(&result.view());
        Ok(())
    }

    /// Inverse 3D WHT: applies forward WHT then divides by N³.
    ///
    /// Correctness: `WHT_3D(WHT_3D(X)) = N³·X`, so scaling by `1/N³` recovers X.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `input.shape() != (n, n, n)`.
    pub fn inverse(&self, input: &Array3<f64>) -> Result<Array3<f64>, FwhtError> {
        let n = self.n;
        let mut result = self.forward(input)?;
        let scale = 1.0 / (n * n * n) as f64;
        scale_array(&mut result, scale);
        Ok(result)
    }

    /// Inverse 3D WHT into a caller-owned output buffer.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when dimensions do not match the plan.
    pub fn inverse_into(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
    ) -> Result<(), FwhtError> {
        let result = self.inverse(input)?;
        output.assign(&result.view());
        Ok(())
    }

    /// Forward 3D WHT of a complex-valued `n×n×n` array.
    ///
    /// Applies the real WHT butterfly independently to real and imaginary parts.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `input.shape() != (n, n, n)`.
    pub fn forward_complex(
        &self,
        input: &Array3<Complex64>,
    ) -> Result<Array3<Complex64>, FwhtError> {
        let n = self.n;
        let [d0, d1, d2] = input.shape();
        if d0 != n || d1 != n || d2 != n {
            return Err(FwhtError::LengthMismatch);
        }
        let re_in: Array3<f64> = input.mapv(|v| v.re);
        let im_in: Array3<f64> = input.mapv(|v| v.im);
        let re_out = self.forward(&re_in)?;
        let im_out = self.forward(&im_in)?;
        Ok(re_out.zip_map(&im_out, Complex64::new))
    }

    /// Inverse 3D WHT of a complex-valued `n×n×n` spectrum.
    ///
    /// # Errors
    /// Returns [`FwhtError::LengthMismatch`] when `input.shape() != (n, n, n)`.
    pub fn inverse_complex(
        &self,
        input: &Array3<Complex64>,
    ) -> Result<Array3<Complex64>, FwhtError> {
        let n = self.n;
        let [d0, d1, d2] = input.shape();
        if d0 != n || d1 != n || d2 != n {
            return Err(FwhtError::LengthMismatch);
        }
        let re_in: Array3<f64> = input.mapv(|v| v.re);
        let im_in: Array3<f64> = input.mapv(|v| v.im);
        let re_out = self.inverse(&re_in)?;
        let im_out = self.inverse(&im_in)?;
        Ok(re_out.zip_map(&im_out, Complex64::new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use leto::Array3;

    #[test]
    fn plan_new_rejects_zero_and_non_power_of_two() {
        assert_eq!(FwhtPlan3D::new(0).unwrap_err(), FwhtError::EmptyInput);
        assert_eq!(FwhtPlan3D::new(3).unwrap_err(), FwhtError::NonPowerOfTwo);
        assert!(FwhtPlan3D::new(2).is_ok());
    }

    /// WHT_3D(WHT_3D(X)) = N³·X (involution).
    #[test]
    fn forward_3d_involution_equals_n_cubed_times_input() {
        let n = 2_usize;
        let plan = FwhtPlan3D::new(n).expect("plan");
        let flat = [1.0_f64, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5];
        let mut input = Array3::<f64>::zeros([n, n, n]);
        let mut idx = 0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    input[[i, j, k]] = flat[idx];
                    idx += 1;
                }
            }
        }
        let first = plan.forward(&input).expect("first forward");
        let second = plan.forward(&first).expect("second forward");
        let n3 = (n * n * n) as f64;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    assert_abs_diff_eq!(
                        second[[i, j, k]],
                        n3 * input[[i, j, k]],
                        epsilon = 1.0e-10
                    );
                }
            }
        }
    }

    /// inverse_3d(forward_3d(X)) = X for all real inputs.
    #[test]
    fn inverse_3d_roundtrip_recovers_signal() {
        let n = 2_usize;
        let plan = FwhtPlan3D::new(n).expect("plan");
        let flat = [1.0_f64, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5];
        let mut input = Array3::<f64>::zeros([n, n, n]);
        let mut idx = 0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    input[[i, j, k]] = flat[idx];
                    idx += 1;
                }
            }
        }
        let spectrum = plan.forward(&input).expect("forward");
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    assert_abs_diff_eq!(recovered[[i, j, k]], input[[i, j, k]], epsilon = 1.0e-12);
                }
            }
        }
    }

    /// 3D WHT of a separable signal: outer product x⊗y⊗z.
    /// W_{3D}(x⊗y⊗z) = WHT(x) ⊗ WHT(y) ⊗ WHT(z) by separability.
    #[test]
    fn forward_3d_matches_separable_outer_product() {
        use super::super::dimension_1d::FwhtPlan;
        use leto::Array1;
        let n = 2_usize;
        let plan_3d = FwhtPlan3D::new(n).expect("3D plan");
        let plan_1d = FwhtPlan::new(n).expect("1D plan");
        let a = Array1::from(vec![1.0_f64, -2.0]);
        let b = Array1::from(vec![3.0_f64, -1.5]);
        let c = Array1::from(vec![0.5_f64, 0.25]);
        let mut input = Array3::<f64>::zeros([n, n, n]);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    input[[i, j, k]] = a[i] * b[j] * c[k];
                }
            }
        }
        let result = plan_3d.forward(&input).expect("3D forward");
        let a_wht = plan_1d.forward(&a).expect("1D a WHT");
        let b_wht = plan_1d.forward(&b).expect("1D b WHT");
        let c_wht = plan_1d.forward(&c).expect("1D c WHT");
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let expected = a_wht[i] * b_wht[j] * c_wht[k];
                    assert_abs_diff_eq!(result[[i, j, k]], expected, epsilon = 1.0e-10);
                }
            }
        }
    }

    #[test]
    fn forward_3d_rejects_non_cubic_input() {
        let plan = FwhtPlan3D::new(2).expect("plan");
        let wrong = Array3::<f64>::zeros([2, 2, 4]);
        assert_eq!(plan.forward(&wrong).unwrap_err(), FwhtError::LengthMismatch);
    }

    #[test]
    fn forward_complex_3d_roundtrip_recovers_signal() {
        let n = 2_usize;
        let plan = FwhtPlan3D::new(n).expect("plan");
        let mut input = Array3::<Complex64>::zeros([n, n, n]);
        input[[0, 0, 0]] = Complex64::new(1.0, 0.5);
        input[[0, 0, 1]] = Complex64::new(-1.0, 2.0);
        input[[0, 1, 0]] = Complex64::new(0.25, -0.75);
        input[[0, 1, 1]] = Complex64::new(3.0, -1.5);
        input[[1, 0, 0]] = Complex64::new(-2.0, 1.0);
        input[[1, 0, 1]] = Complex64::new(0.5, 0.25);
        input[[1, 1, 0]] = Complex64::new(-0.5, 2.1);
        input[[1, 1, 1]] = Complex64::new(1.5, -0.2);
        let spectrum = plan.forward_complex(&input).expect("complex forward");
        let recovered = plan.inverse_complex(&spectrum).expect("complex inverse");
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    assert_abs_diff_eq!(
                        recovered[[i, j, k]].re,
                        input[[i, j, k]].re,
                        epsilon = 1.0e-12
                    );
                    assert_abs_diff_eq!(
                        recovered[[i, j, k]].im,
                        input[[i, j, k]].im,
                        epsilon = 1.0e-12
                    );
                }
            }
        }
    }
}
