//! Wavelet coefficient containers.

use leto::Array2;

/// Multilevel DWT coefficient storage.
#[derive(Debug, Clone, PartialEq)]
pub struct DwtCoefficients {
    len: usize,
    levels: usize,
    approximation: Vec<f64>,
    details: Vec<Vec<f64>>,
}

impl DwtCoefficients {
    /// Create DWT coefficient storage.
    #[must_use]
    pub fn new(len: usize, levels: usize, approximation: Vec<f64>, details: Vec<Vec<f64>>) -> Self {
        Self {
            len,
            levels,
            approximation,
            details,
        }
    }

    /// Return original signal length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return true when original signal length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return decomposition levels.
    #[must_use]
    pub const fn levels(&self) -> usize {
        self.levels
    }

    /// Return the coarsest approximation coefficients.
    #[must_use]
    pub fn approximation(&self) -> &[f64] {
        &self.approximation
    }

    /// Return detail coefficients from finest to coarsest.
    #[must_use]
    pub fn details(&self) -> &[Vec<f64>] {
        &self.details
    }
}

/// CWT coefficient matrix with shape `(scales, samples)`.
#[derive(Debug, Clone, PartialEq)]
pub struct CwtCoefficients {
    scales: Vec<f64>,
    values: Array2<f64>,
}

impl CwtCoefficients {
    /// Create CWT coefficient storage.
    #[must_use]
    pub fn new(scales: Vec<f64>, values: Array2<f64>) -> Self {
        Self { scales, values }
    }

    /// Return CWT scales.
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Return dense coefficient matrix.
    #[must_use]
    pub fn values(&self) -> &Array2<f64> {
        &self.values
    }
}
