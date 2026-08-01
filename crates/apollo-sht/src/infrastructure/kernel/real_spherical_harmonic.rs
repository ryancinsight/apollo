//! Real spherical-harmonic basis for diffusion MRI.
//!
//! Diffusion signals are antipodally symmetric (`S(g) = S(-g)`), so only
//! even-degree spherical harmonics contribute. This module provides a single
//! real basis-function evaluator and even-order basis metadata for scattered
//! gradient directions.
//!
//! # Real basis convention
//!
//! From the complex, orthonormal `Y_l^m` with the Condon-Shortley phase:
//!
//! ```text
//! R_l^0(theta, phi)    = Y_l^0(theta, phi)
//! R_l^m(theta, phi)    = sqrt(2) Re(Y_l^m(theta, phi))  for m > 0
//! R_l^-m(theta, phi)   = sqrt(2) Im(Y_l^m(theta, phi))  for m > 0
//! ```
//!
//! This is the orthonormal convention documented by
//! [MRtrix3](https://mrtrix.readthedocs.io/en/dev/concepts/spherical_harmonics.html#formulation-used-in-mrtrix3).
//! The basis satisfies
//! `integral R_l^m R_l'^m' dOmega = delta(l,l') delta(m,m')` on the unit sphere.
//!
//! Only degrees `0, 2, 4, ..., l_max` are included in
//! [`crate::RealSphericalHarmonicBasis`]. The coefficient count is
//! `(l_max + 1)(l_max + 2) / 2`.

use thiserror::Error as ThisError;

use super::spherical_harmonic::spherical_harmonic;

/// Largest degree whose worst-case normalization product, `(2l)!`, remains
/// finite in binary64 (`170!` is finite and `172!` is not).
pub const MAX_REAL_SH_DEGREE: usize = 85;

const UNIT_NORM_SQUARED_TOLERANCE: f64 = 32.0 * f64::EPSILON;

/// Error returned by real spherical-harmonic basis operations.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, ThisError)]
pub enum RealShError {
    /// `l_max` is not even.
    #[error("l_max must be even for an antipodally symmetric basis, got {0}")]
    OddLMax(usize),
    /// `l_max` is less than two.
    #[error("l_max must be at least 2, got {0}")]
    TooSmall(usize),
    /// The requested degree exceeds the stable binary64 range.
    #[error("degree {degree} exceeds the supported maximum {maximum}")]
    DegreeOutOfRange {
        /// Requested harmonic degree.
        degree: usize,
        /// Maximum supported degree.
        maximum: usize,
    },
    /// The coefficient-count formula overflowed `usize`.
    #[error("coefficient count overflows usize for l_max {0}")]
    CoefficientCountOverflow(usize),
    /// Matrix element-count arithmetic overflowed `usize`.
    #[error("matrix element count overflows usize for {rows} rows and {columns} columns")]
    MatrixSizeOverflow {
        /// Matrix row count.
        rows: usize,
        /// Matrix column count.
        columns: usize,
    },
    /// Storage for the basis or an evaluated matrix could not be reserved.
    #[error("could not reserve storage for {element_count} spherical-harmonic values")]
    AllocationFailed {
        /// Number of values requested.
        element_count: usize,
    },
    /// A harmonic order lies outside the degree's valid interval.
    #[error(
        "real spherical harmonic requires |order| <= degree, got degree {degree}, order {order}"
    )]
    InvalidOrder {
        /// Harmonic degree.
        degree: usize,
        /// Signed harmonic order.
        order: isize,
    },
    /// The polar angle is non-finite or outside `[0, pi]`.
    #[error("polar angle theta must be finite and in [0, pi], got {0}")]
    InvalidTheta(f64),
    /// The azimuthal angle is non-finite.
    #[error("azimuthal angle phi must be finite, got {0}")]
    InvalidPhi(f64),
    /// A Cartesian direction contains a non-finite component.
    #[error("direction component {axis} must be finite, got {value}")]
    NonFiniteDirection {
        /// Cartesian axis index.
        axis: usize,
        /// Non-finite component value.
        value: f64,
    },
    /// A Cartesian direction does not have unit length within the rounding bound.
    #[error("direction squared norm must equal one within {tolerance}, got {norm_squared}")]
    NonUnitDirection {
        /// Computed squared norm.
        norm_squared: f64,
        /// Accepted absolute squared-norm error.
        tolerance: f64,
    },
    /// Evaluation produced a non-finite basis value.
    #[error("real spherical harmonic evaluation is not finite for degree {degree}, order {order}")]
    NonFiniteEvaluation {
        /// Harmonic degree.
        degree: usize,
        /// Signed harmonic order.
        order: isize,
    },
    /// Leto rejected completed row-major matrix storage.
    #[error("Leto rejected a {rows} by {columns} real spherical-harmonic matrix")]
    MatrixShape {
        /// Matrix row count.
        rows: usize,
        /// Matrix column count.
        columns: usize,
    },
}

/// Metadata and evaluation for a real, even-order, orthonormal spherical-
/// harmonic basis.
///
/// Coefficients use degree-major, order-minor ordering over even degrees:
/// `l=0: m=0`, `l=2: m=-2..=2`, `l=4: m=-4..=4`, and so on.
#[derive(Clone, Debug)]
pub struct RealSphericalHarmonicBasis {
    l_max: usize,
    lm_table: Vec<(usize, isize)>,
}

impl RealSphericalHarmonicBasis {
    /// Create a basis for even degrees `0, 2, 4, ..., l_max`.
    ///
    /// # Errors
    ///
    /// Returns a typed error if `l_max` is odd, less than two, exceeds
    /// [`MAX_REAL_SH_DEGREE`], overflows sizing arithmetic, or cannot be
    /// allocated.
    pub fn new(l_max: usize) -> Result<Self, RealShError> {
        if l_max < 2 {
            return Err(RealShError::TooSmall(l_max));
        }
        if l_max % 2 != 0 {
            return Err(RealShError::OddLMax(l_max));
        }
        validate_degree(l_max)?;

        let coefficient_count = l_max
            .checked_add(1)
            .and_then(|value| value.checked_mul(l_max.checked_add(2)?))
            .map(|value| value / 2)
            .ok_or(RealShError::CoefficientCountOverflow(l_max))?;
        let mut lm_table = Vec::new();
        lm_table.try_reserve_exact(coefficient_count).map_err(|_| {
            RealShError::AllocationFailed {
                element_count: coefficient_count,
            }
        })?;

        for degree in (0..=l_max).step_by(2) {
            let signed_degree =
                isize::try_from(degree).map_err(|_| RealShError::DegreeOutOfRange {
                    degree,
                    maximum: MAX_REAL_SH_DEGREE,
                })?;
            for order in -signed_degree..=signed_degree {
                lm_table.push((degree, order));
            }
        }
        debug_assert_eq!(lm_table.len(), coefficient_count);

        Ok(Self { l_max, lm_table })
    }

    /// Maximum even degree.
    #[must_use]
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    /// Number of real spherical-harmonic coefficients.
    #[must_use]
    pub fn num_coefficients(&self) -> usize {
        self.lm_table.len()
    }

    /// Map a flattened coefficient index to its `(degree, order)` pair.
    #[must_use]
    pub fn index_to_lm(&self, index: usize) -> Option<(usize, isize)> {
        self.lm_table.get(index).copied()
    }

    /// Evaluate every basis function at `(theta, phi)` in canonical order.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid angles, non-finite evaluation, or
    /// allocation failure.
    pub fn evaluate(&self, theta: f64, phi: f64) -> Result<Vec<f64>, RealShError> {
        validate_angles(theta, phi)?;
        let mut values = Vec::new();
        values.try_reserve_exact(self.lm_table.len()).map_err(|_| {
            RealShError::AllocationFailed {
                element_count: self.lm_table.len(),
            }
        })?;
        for &(degree, order) in &self.lm_table {
            values.push(real_spherical_harmonic(degree, order, theta, phi)?);
        }
        Ok(values)
    }

    /// Evaluate every basis function at a Cartesian unit direction.
    ///
    /// # Errors
    ///
    /// Returns a typed error if any component is non-finite or the squared
    /// norm differs from one by more than `32 * f64::EPSILON`. That bound
    /// covers the three products and two additions used to check a vector
    /// normalized in binary64 arithmetic.
    pub fn evaluate_at_direction(&self, direction: &[f64; 3]) -> Result<Vec<f64>, RealShError> {
        let (theta, phi) = cartesian_to_spherical(direction)?;
        self.evaluate(theta, phi)
    }

    /// Build `B[i, k] = R_l(k)^m(k)(direction_i)` as one row-major allocation.
    ///
    /// The operation costs `O(N * K * l_max)` for `N` directions and `K`
    /// coefficients. No temporary row vectors are allocated.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid directions, non-finite evaluation,
    /// size overflow, allocation failure, or matrix construction failure.
    pub fn design_matrix(&self, directions: &[[f64; 3]]) -> Result<leto::Array2<f64>, RealShError> {
        let rows = directions.len();
        let columns = self.lm_table.len();
        let element_count = rows
            .checked_mul(columns)
            .ok_or(RealShError::MatrixSizeOverflow { rows, columns })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(element_count)
            .map_err(|_| RealShError::AllocationFailed { element_count })?;

        for direction in directions {
            let (theta, phi) = cartesian_to_spherical(direction)?;
            for &(degree, order) in &self.lm_table {
                values.push(real_spherical_harmonic(degree, order, theta, phi)?);
            }
        }

        leto::Array2::from_vec([rows, columns], values)
            .map_err(|_| RealShError::MatrixShape { rows, columns })
    }

    /// Iterate over `(index, degree, order)` triples in coefficient order.
    pub fn iter_lm(&self) -> impl Iterator<Item = (usize, usize, isize)> + '_ {
        self.lm_table
            .iter()
            .enumerate()
            .map(|(index, &(degree, order))| (index, degree, order))
    }
}

/// Evaluate one real orthonormal spherical harmonic `R_l^m(theta, phi)`.
///
/// # Errors
///
/// Returns a typed error if the degree exceeds [`MAX_REAL_SH_DEGREE`],
/// `|order| > degree`, either angle lies outside its finite domain, or the
/// evaluated value is non-finite.
pub fn real_spherical_harmonic(
    degree: usize,
    order: isize,
    theta: f64,
    phi: f64,
) -> Result<f64, RealShError> {
    validate_degree(degree)?;
    if order.unsigned_abs() > degree {
        return Err(RealShError::InvalidOrder { degree, order });
    }
    validate_angles(theta, phi)?;

    let harmonic = spherical_harmonic(degree, order.unsigned_abs() as isize, theta, phi);
    let value = match order.cmp(&0) {
        std::cmp::Ordering::Equal => harmonic.re,
        std::cmp::Ordering::Greater => std::f64::consts::SQRT_2 * harmonic.re,
        std::cmp::Ordering::Less => std::f64::consts::SQRT_2 * harmonic.im,
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RealShError::NonFiniteEvaluation { degree, order })
    }
}

fn validate_degree(degree: usize) -> Result<(), RealShError> {
    if degree > MAX_REAL_SH_DEGREE {
        return Err(RealShError::DegreeOutOfRange {
            degree,
            maximum: MAX_REAL_SH_DEGREE,
        });
    }
    Ok(())
}

fn validate_angles(theta: f64, phi: f64) -> Result<(), RealShError> {
    if !theta.is_finite() || !(0.0..=std::f64::consts::PI).contains(&theta) {
        return Err(RealShError::InvalidTheta(theta));
    }
    if !phi.is_finite() {
        return Err(RealShError::InvalidPhi(phi));
    }
    Ok(())
}

fn cartesian_to_spherical(direction: &[f64; 3]) -> Result<(f64, f64), RealShError> {
    for (axis, &value) in direction.iter().enumerate() {
        if !value.is_finite() {
            return Err(RealShError::NonFiniteDirection { axis, value });
        }
    }

    let [x, y, z] = *direction;
    let norm_squared = x.mul_add(x, y.mul_add(y, z * z));
    if !norm_squared.is_finite() || (norm_squared - 1.0).abs() > UNIT_NORM_SQUARED_TOLERANCE {
        return Err(RealShError::NonUnitDirection {
            norm_squared,
            tolerance: UNIT_NORM_SQUARED_TOLERANCE,
        });
    }

    let theta = z.clamp(-1.0, 1.0).acos();
    let phi = f64::atan2(y, x);
    let phi = if phi < 0.0 {
        phi + std::f64::consts::TAU
    } else {
        phi
    };
    Ok((theta, phi))
}

#[cfg(test)]
mod tests;
