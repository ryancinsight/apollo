use eunomia::RealField;

use crate::transform::coefficient;
use crate::{Normalization, PlanError};

/// Precomputed fixed-capacity Type-III discrete cosine transform.
///
/// The plan stores its complete `N × N` basis inline and allocates no heap
/// memory. Its size is exactly `N * N * size_of::<T>()` bytes, so callers choose
/// `N` according to their stack-capacity budget.
#[derive(Clone, Debug, PartialEq)]
pub struct DctIiiPlan<T, const N: usize> {
    basis: [[T; N]; N],
}

impl<T: RealField, const N: usize> DctIiiPlan<T, N> {
    /// Construct a plan with the selected normalization.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError::EmptyLength`] when `N` is zero.
    pub fn new(normalization: Normalization) -> Result<Self, PlanError> {
        if N == 0 {
            return Err(PlanError::EmptyLength);
        }
        let mut basis = [[T::ZERO; N]; N];
        for (position, row) in basis.iter_mut().enumerate() {
            for (frequency, value) in row.iter_mut().enumerate() {
                *value = coefficient(normalization, N, position, frequency);
            }
        }
        Ok(Self { basis })
    }

    /// Apply the precomputed transform into caller-owned output.
    ///
    /// Arithmetic follows the [`RealField`] implementation. NaNs propagate
    /// through affected sums, infinities follow ordinary IEEE-754 operations,
    /// and exact zero sums may have a positive sign because accumulation starts
    /// at `+0`.
    pub fn transform(&self, signal: &[T; N], output: &mut [T; N]) {
        for (slot, row) in output.iter_mut().zip(&self.basis) {
            *slot = signal
                .iter()
                .copied()
                .zip(row.iter().copied())
                .fold(T::ZERO, |sum, (sample, weight)| sum + sample * weight);
        }
    }
}

#[cfg(test)]
mod tests;
