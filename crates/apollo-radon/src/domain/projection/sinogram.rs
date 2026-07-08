//! Dense sinogram storage.

use leto::Array2;

/// Dense sinogram indexed by `(angle_index, detector_index)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sinogram {
    values: Array2<f64>,
}

impl Sinogram {
    /// Create sinogram storage from computed projection values.
    #[must_use]
    pub fn new(values: Array2<f64>) -> Self {
        Self { values }
    }

    /// Borrow sinogram values.
    #[must_use]
    pub const fn values(&self) -> &Array2<f64> {
        &self.values
    }

    /// Consume storage and return projection values.
    #[must_use]
    pub fn into_values(self) -> Array2<f64> {
        self.values
    }

    /// Return `(angle_count, detector_count)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        let [angles, detectors] = self.values.shape();
        (angles, detectors)
    }
}
