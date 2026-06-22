//! WGPU plan descriptors.

/// Metadata-preserving WGPU plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RadonWgpuPlan {
    rows: usize,
    cols: usize,
    angle_count: usize,
    detector_count: usize,
    detector_spacing_bits: u64,
}

impl RadonWgpuPlan {
    /// Create a WGPU plan descriptor carrying the Radon geometry shape.
    #[must_use]
    pub const fn new(
        rows: usize,
        cols: usize,
        angle_count: usize,
        detector_count: usize,
        detector_spacing_bits: u64,
    ) -> Self {
        Self {
            rows,
            cols,
            angle_count,
            detector_count,
            detector_spacing_bits,
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

    /// Return whether the descriptor carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.rows == 0 || self.cols == 0 || self.angle_count == 0 || self.detector_count == 0
    }
}
