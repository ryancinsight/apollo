//! WGPU plan descriptors.

/// Metadata-preserving WGPU plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShtWgpuPlan {
    latitudes: usize,
    longitudes: usize,
    max_degree: usize,
}

impl ShtWgpuPlan {
    /// Create a WGPU plan descriptor for a spherical grid and bandlimit.
    #[must_use]
    pub const fn new(latitudes: usize, longitudes: usize, max_degree: usize) -> Self {
        Self {
            latitudes,
            longitudes,
            max_degree,
        }
    }

    /// Return the latitude sample count.
    #[must_use]
    pub const fn latitudes(self) -> usize {
        self.latitudes
    }

    /// Return the longitude sample count.
    #[must_use]
    pub const fn longitudes(self) -> usize {
        self.longitudes
    }

    /// Return the maximum spherical harmonic degree.
    #[must_use]
    pub const fn max_degree(self) -> usize {
        self.max_degree
    }

    /// Return the number of grid samples.
    #[must_use]
    pub const fn sample_count(self) -> usize {
        self.latitudes * self.longitudes
    }

    /// Return the number of valid `(degree, order)` modes.
    #[must_use]
    pub const fn mode_count(self) -> usize {
        let degree_count = self.max_degree + 1;
        degree_count * degree_count
    }

    /// Return whether the descriptor carries an invalid empty shape.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.latitudes == 0 || self.longitudes == 0
    }
}
