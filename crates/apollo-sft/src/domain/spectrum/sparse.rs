//! Sparse spectral representation.

use apollo_fft::{ApolloError, ApolloResult};
use eunomia::Complex64;

/// Sparse spectral representation of a complex-valued signal.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseSpectrum {
    /// Signal length.
    pub n: usize,
    /// Recovered frequency bins in ascending order.
    pub frequencies: Vec<usize>,
    /// Complex coefficients aligned with [`Self::frequencies`].
    pub values: Vec<Complex64>,
}

impl SparseSpectrum {
    /// Create an empty sparse spectrum for a signal of length `n`.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            frequencies: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> ApolloResult<()> {
        if self.frequencies.len() != self.values.len() {
            return Err(ApolloError::validation(
                "frequencies",
                self.frequencies.len().to_string(),
                "frequencies and values must have identical length",
            ));
        }
        if self.frequencies.iter().any(|&freq| freq >= self.n) {
            return Err(ApolloError::validation(
                "frequency",
                self.frequencies
                    .iter()
                    .copied()
                    .find(|&freq| freq >= self.n)
                    .unwrap_or_default()
                    .to_string(),
                "frequency must be strictly less than the signal length",
            ));
        }
        if self
            .frequencies
            .windows(2)
            .any(|pair| matches!(pair, [a, b] if a >= b))
        {
            return Err(ApolloError::validation(
                "frequencies",
                "non-ascending",
                "frequencies must be strictly ascending",
            ));
        }
        Ok(())
    }

    /// Insert or replace a coefficient.
    pub fn insert(&mut self, frequency: usize, value: Complex64) -> ApolloResult<()> {
        if frequency >= self.n {
            return Err(ApolloError::validation(
                "frequency",
                frequency.to_string(),
                "frequency must be strictly less than the signal length",
            ));
        }

        match self.frequencies.binary_search(&frequency) {
            Ok(index) => self.values[index] = value,
            Err(index) => {
                self.frequencies.insert(index, frequency);
                self.values.insert(index, value);
            }
        }

        Ok(())
    }

    /// Return the retained support size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frequencies.len()
    }

    /// Return an iterator of (frequency_index, coefficient) pairs in ascending
    /// frequency order. Both types are Copy; the iterator yields owned pairs.
    pub fn coefficients(&self) -> impl Iterator<Item = (usize, Complex64)> + use<'_> {
        self.frequencies
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }
    /// Whether the spectrum stores no coefficients.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frequencies.is_empty()
    }

    /// Expand into a dense coefficient vector of length `n`.
    #[must_use]
    pub fn to_dense(&self) -> Vec<Complex64> {
        let mut dense = vec![Complex64::new(0.0, 0.0); self.n];
        for (&frequency, &value) in self.frequencies.iter().zip(self.values.iter()) {
            dense[frequency] = value;
        }
        dense
    }
}
