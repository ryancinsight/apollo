//! Dense spherical harmonic coefficient storage.
//!
//! Coefficients are stored by degree `l` and signed order `m`. The backing
//! matrix has shape `(max_degree + 1, 2 * max_degree + 1)`, with order index
//! `m + max_degree`. Entries where `|m| > l` are invalid and left
//! at zero by transform construction.

use leto::Array2;
use eunomia::Complex64;

/// Dense spherical harmonic coefficient matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalHarmonicCoefficients {
    max_degree: usize,
    values: Array2<Complex64>,
}

impl SphericalHarmonicCoefficients {
    /// Create a zero-filled coefficient matrix for a bandlimit.
    #[must_use]
    pub fn zeros(max_degree: usize) -> Self {
        Self {
            max_degree,
            values: Array2::zeros([max_degree + 1, 2 * max_degree + 1]),
        }
    }

    /// Create coefficient storage from a dense matrix.
    #[must_use]
    pub fn from_values(max_degree: usize, values: Array2<Complex64>) -> Self {
        Self { max_degree, values }
    }

    /// Return maximum spherical harmonic degree.
    #[must_use]
    pub const fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Return the dense backing matrix.
    #[must_use]
    pub fn values(&self) -> &Array2<Complex64> {
        &self.values
    }

    /// Return mutable dense backing matrix.
    #[must_use]
    pub fn values_mut(&mut self) -> &mut Array2<Complex64> {
        &mut self.values
    }

    /// Return the coefficient for degree `l` and signed order `m`.
    #[must_use]
    pub fn get(&self, degree: usize, order: isize) -> Complex64 {
        self.values[[degree, self.order_index(order)]]
    }

    /// Set the coefficient for degree `l` and signed order `m`.
    pub fn set(&mut self, degree: usize, order: isize, value: Complex64) {
        let order_index = self.order_index(order);
        self.values[[degree, order_index]] = value;
    }

    fn order_index(&self, order: isize) -> usize {
        (order + self.max_degree as isize) as usize
    }
}
