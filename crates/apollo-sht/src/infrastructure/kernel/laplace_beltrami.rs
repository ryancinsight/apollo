//! Laplace-Beltrami regularization for spherical-harmonic fitting.
//!
//! # Why fitting a real SH basis needs regularization
//!
//! Recovering coefficients `c` from samples `s` measured along `N` directions
//! means solving `B c ≈ s`, where `B` is the design matrix from
//! [`crate::RealSphericalHarmonicBasis::design_matrix`].
//! The problem is ill-conditioned in practice: an order-8 basis carries 45
//! coefficients, a typical acquisition supplies 60 directions, and the
//! directions are not a quadrature grid. Least squares alone then fits noise
//! into the high-degree coefficients, where the basis oscillates fastest.
//!
//! The standard remedy penalizes the *roughness* of the reconstructed function
//! rather than the size of its coefficients, so a genuinely sharp feature
//! survives while noise-driven oscillation does not.
//!
//! # The operator
//!
//! The Laplace-Beltrami operator `Δ_b` is the Laplacian restricted to the
//! sphere, and the spherical harmonics are its eigenfunctions:
//!
//! ```text
//! Δ_b Y_l^m = -l(l + 1) Y_l^m
//! ```
//!
//! The roughness functional `∫_{S²} (Δ_b f)² dΩ` therefore diagonalizes in the
//! SH basis. For `f = Σ c_i R_{l_i}^{m_i}` with an orthonormal basis,
//!
//! ```text
//! ∫ (Δ_b f)² dΩ = Σ_i c_i² l_i²(l_i + 1)²  =  cᵀ L c
//! ```
//!
//! so `L` is diagonal with entries `l²(l + 1)²`, depending only on the degree.
//! The regularized fit solves
//!
//! ```text
//! (BᵀB + λL) c = Bᵀs
//! ```
//!
//! Degree zero carries a zero penalty, so the isotropic component — the mean
//! signal — is never shrunk. The penalty then grows as `l⁴`, which is what
//! makes the operator discriminate sharply between smooth and oscillatory
//! content instead of damping everything alike.
//!
//! # Reference
//!
//! Descoteaux, Angelino, Fitzgibbons, and Deriche, "Regularized, fast, and
//! robust analytical Q-ball imaging", *Magnetic Resonance in Medicine* 58(3),
//! 2007, §2.3 — the formulation this module implements, including the
//! `l²(l+1)²` diagonal and its use in the normal equations above.

use super::real_spherical_harmonic::{RealShError, RealSphericalHarmonicBasis};

impl RealSphericalHarmonicBasis {
    /// Diagonal entries of the Laplace-Beltrami penalty `L`, in coefficient
    /// order.
    ///
    /// Entry `i` is `l_i²(l_i + 1)²`, the squared eigenvalue of `Δ_b` for that
    /// coefficient's degree. Callers assembling their own normal equations use
    /// this directly; [`Self::laplace_beltrami_matrix`] is the same values as a
    /// square matrix.
    ///
    /// # Errors
    ///
    /// [`RealShError::AllocationFailed`] if storage cannot be reserved.
    pub fn laplace_beltrami_diagonal(&self) -> Result<Vec<f64>, RealShError> {
        let count = self.num_coefficients();
        let mut diagonal = Vec::new();
        diagonal
            .try_reserve_exact(count)
            .map_err(|_| RealShError::AllocationFailed {
                element_count: count,
            })?;

        for (_, degree, _) in self.iter_lm() {
            diagonal.push(laplace_beltrami_eigenvalue(degree));
        }
        Ok(diagonal)
    }

    /// The Laplace-Beltrami penalty matrix `L`, square and diagonal.
    ///
    /// Supplied as a matrix for callers that assemble `BᵀB + λL` through dense
    /// linear algebra. A caller that can add a diagonal in place should prefer
    /// [`Self::laplace_beltrami_diagonal`] and avoid the `K²` allocation.
    ///
    /// # Errors
    ///
    /// [`RealShError::MatrixSizeOverflow`] if `K²` overflows `usize`,
    /// [`RealShError::AllocationFailed`] if storage cannot be reserved, or
    /// [`RealShError::MatrixShape`] if matrix construction fails.
    pub fn laplace_beltrami_matrix(&self) -> Result<leto::Array2<f64>, RealShError> {
        let count = self.num_coefficients();
        let element_count = count
            .checked_mul(count)
            .ok_or(RealShError::MatrixSizeOverflow {
                rows: count,
                columns: count,
            })?;

        let mut values = Vec::new();
        values
            .try_reserve_exact(element_count)
            .map_err(|_| RealShError::AllocationFailed { element_count })?;
        values.resize(element_count, 0.0);

        for (index, degree, _) in self.iter_lm() {
            values[index * count + index] = laplace_beltrami_eigenvalue(degree);
        }

        leto::Array2::from_vec([count, count], values).map_err(|_| RealShError::MatrixShape {
            rows: count,
            columns: count,
        })
    }
}

/// The squared Laplace-Beltrami eigenvalue `l²(l + 1)²` for `degree`.
///
/// `Δ_b Y_l^m = -l(l+1) Y_l^m`, so penalizing `∫(Δ_b f)²` weights each
/// coefficient by the square of that eigenvalue. Degree zero yields zero: the
/// isotropic component is not penalized.
///
/// Computed in `f64` from the exact integer product, which is representable
/// without rounding for every degree the basis admits.
#[must_use]
pub fn laplace_beltrami_eigenvalue(degree: usize) -> f64 {
    let l = degree as f64;
    let eigenvalue = l * (l + 1.0);
    eigenvalue * eigenvalue
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigenvalue_matches_the_closed_form() {
        // l²(l+1)²: 0, 4²=... — computed by hand from l(l+1) squared.
        assert_eq!(laplace_beltrami_eigenvalue(0), 0.0);
        assert_eq!(laplace_beltrami_eigenvalue(2), 36.0); // (2·3)² = 36
        assert_eq!(laplace_beltrami_eigenvalue(4), 400.0); // (4·5)² = 400
        assert_eq!(laplace_beltrami_eigenvalue(6), 1764.0); // (6·7)² = 1764
        assert_eq!(laplace_beltrami_eigenvalue(8), 5184.0); // (8·9)² = 5184
    }

    #[test]
    fn isotropic_component_is_never_penalized() {
        // Degree zero is the mean signal. Shrinking it would bias every fitted
        // amplitude toward zero, which is not what roughness regularization is
        // for.
        let basis = RealSphericalHarmonicBasis::new(8).expect("valid basis");
        let diagonal = basis.laplace_beltrami_diagonal().expect("diagonal");
        assert_eq!(
            diagonal[0], 0.0,
            "the l=0 coefficient carries no roughness penalty"
        );
    }

    #[test]
    fn diagonal_is_constant_within_a_degree() {
        // The operator depends on degree alone, so every order of one degree
        // must share a penalty; an order-dependent entry would make the
        // regularizer prefer particular orientations.
        let basis = RealSphericalHarmonicBasis::new(6).expect("valid basis");
        let diagonal = basis.laplace_beltrami_diagonal().expect("diagonal");

        for (index, degree, _) in basis.iter_lm() {
            assert_eq!(
                diagonal[index],
                laplace_beltrami_eigenvalue(degree),
                "coefficient {index} of degree {degree} must carry its degree's penalty"
            );
        }
    }

    #[test]
    fn diagonal_length_matches_the_coefficient_count() {
        for l_max in [2, 4, 6, 8] {
            let basis = RealSphericalHarmonicBasis::new(l_max).expect("valid basis");
            assert_eq!(
                basis.laplace_beltrami_diagonal().expect("diagonal").len(),
                basis.num_coefficients(),
                "one penalty per coefficient at l_max {l_max}"
            );
        }
    }

    #[test]
    fn penalty_grows_monotonically_with_degree() {
        // The l⁴ growth is what makes the operator discriminate between smooth
        // and oscillatory content rather than damping uniformly.
        let basis = RealSphericalHarmonicBasis::new(8).expect("valid basis");
        let diagonal = basis.laplace_beltrami_diagonal().expect("diagonal");

        let mut previous_degree = 0;
        let mut previous_penalty = -1.0;
        for (index, degree, _) in basis.iter_lm() {
            if degree != previous_degree {
                assert!(
                    diagonal[index] > previous_penalty,
                    "degree {degree} must be penalized more than degree {previous_degree}"
                );
                previous_degree = degree;
                previous_penalty = diagonal[index];
            }
        }
    }

    #[test]
    fn matrix_is_the_diagonal_embedded_in_a_square() {
        let basis = RealSphericalHarmonicBasis::new(4).expect("valid basis");
        let diagonal = basis.laplace_beltrami_diagonal().expect("diagonal");
        let matrix = basis.laplace_beltrami_matrix().expect("matrix");
        let count = basis.num_coefficients();

        assert_eq!(matrix.shape(), [count, count]);
        let values = matrix.as_slice().expect("contiguous matrix");
        for row in 0..count {
            for column in 0..count {
                let expected = if row == column { diagonal[row] } else { 0.0 };
                assert_eq!(
                    values[row * count + column],
                    expected,
                    "entry ({row}, {column}) must be {expected}"
                );
            }
        }
    }

    #[test]
    fn penalty_equals_the_roughness_of_a_single_harmonic() {
        // For f = R_l^m, the roughness functional integral(Delta_b f)^2 reduces
        // to l^2(l+1)^2 by orthonormality. Reproducing that from the diagonal
        // is the oracle tying the matrix to the functional it represents.
        let basis = RealSphericalHarmonicBasis::new(6).expect("valid basis");
        let diagonal = basis.laplace_beltrami_diagonal().expect("diagonal");

        for (index, degree, _) in basis.iter_lm() {
            // A unit coefficient vector selecting exactly this harmonic.
            let mut coefficients = vec![0.0; basis.num_coefficients()];
            coefficients[index] = 1.0;

            let roughness: f64 = coefficients
                .iter()
                .zip(&diagonal)
                .map(|(coefficient, penalty)| coefficient * coefficient * penalty)
                .sum();

            let expected = (degree * (degree + 1)) as f64;
            assert_eq!(
                roughness,
                expected * expected,
                "roughness of the degree-{degree} harmonic must be l^2(l+1)^2"
            );
        }
    }
}
