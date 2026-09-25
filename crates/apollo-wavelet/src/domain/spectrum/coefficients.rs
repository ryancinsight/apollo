//! Wavelet coefficient containers.

use crate::domain::contracts::error::{WaveletError, WaveletResult};
use leto::Array2;

/// Multilevel DWT coefficient storage.
///
/// Detail levels are stored contiguously in one buffer, finest level first:
/// level `i` occupies `detail_level(i) = &details[len - (len >> i)..len - (len >> (i + 1))]`,
/// the halving shape [`DwtPlan::coefficient_shapes`](crate::application::execution::plan::DwtPlan::coefficient_shapes)
/// fixes for every level — one allocation instead of one per level, and level
/// access is a slice view rather than a pointer chase.
#[derive(Debug, Clone, PartialEq)]
pub struct DwtCoefficients {
    len: usize,
    levels: usize,
    approximation: Vec<f64>,
    details: Vec<f64>,
}

impl DwtCoefficients {
    /// Create DWT coefficient storage from the flat detail buffer.
    ///
    /// # Errors
    /// `CoefficientShapeMismatch` unless `approximation.len() == len >> levels`
    /// and `details.len() == len - (len >> levels)` — the two halves of the
    /// telescoping shape the level slices are derived from.
    pub fn new(
        len: usize,
        levels: usize,
        approximation: Vec<f64>,
        details: Vec<f64>,
    ) -> WaveletResult<Self> {
        if approximation.len() != len >> levels || details.len() != len - (len >> levels) {
            return Err(WaveletError::CoefficientShapeMismatch);
        }
        Ok(Self {
            len,
            levels,
            approximation,
            details,
        })
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

    /// Return the flat detail buffer, finest level first.
    ///
    /// Level boundaries are derived from `(len, levels)`; prefer
    /// [`Self::detail_levels`] for per-level views.
    #[must_use]
    pub fn details(&self) -> &[f64] {
        &self.details
    }

    /// Iterate the detail levels, finest first, each as a contiguous slice.
    pub fn detail_levels(&self) -> impl DoubleEndedIterator<Item = &[f64]> {
        let len = self.len;
        (0..self.levels).map(move |level| self.detail_slice(len, level))
    }

    /// Return detail level `level` (0 = finest), or `None` when out of range.
    #[must_use]
    pub fn detail_level(&self, level: usize) -> Option<&[f64]> {
        (level < self.levels).then(|| self.detail_slice(self.len, level))
    }

    /// Level slice from the telescoping offsets; `level < self.levels` is the
    /// caller's proven invariant.
    fn detail_slice(&self, len: usize, level: usize) -> &[f64] {
        &self.details[len - (len >> level)..len - (len >> (level + 1))]
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
