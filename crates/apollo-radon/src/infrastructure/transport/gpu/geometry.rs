/// Plan payload for the parallel-beam Radon transform: image and
/// sinogram geometry, with the detector spacing stored as an IEEE bit
/// pattern so the payload stays `Eq`-clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryPlan {
    rows: usize,
    cols: usize,
    angle_count: usize,
    detector_count: usize,
    detector_spacing_bits: u64,
}

impl GeometryPlan {
    /// Create a geometry-plan payload.
    #[must_use]
    pub fn new(
        rows: usize,
        cols: usize,
        angle_count: usize,
        detector_count: usize,
        detector_spacing: f64,
    ) -> Self {
        Self {
            rows,
            cols,
            angle_count,
            detector_count,
            detector_spacing_bits: detector_spacing.to_bits(),
        }
    }

    /// Return the image row count.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Return the image column count.
    #[must_use]
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Return the projection angle count.
    #[must_use]
    pub const fn angle_count(self) -> usize {
        self.angle_count
    }

    /// Return the detector bin count.
    #[must_use]
    pub const fn detector_count(self) -> usize {
        self.detector_count
    }

    /// Return the detector spacing.
    #[must_use]
    pub const fn detector_spacing(self) -> f64 {
        f64::from_bits(self.detector_spacing_bits)
    }
}
