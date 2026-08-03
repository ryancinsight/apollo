/// Plan payload for the spherical harmonic transform: grid shape and
/// bandlimit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SphericalPlan {
    latitudes: usize,
    longitudes: usize,
    max_degree: usize,
}

impl SphericalPlan {
    /// Create a spherical-plan payload for a grid and bandlimit.
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
}
