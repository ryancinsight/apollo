//! WGPU plan descriptors.

/// Metadata-preserving WGPU plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MellinWgpuPlan {
    samples: usize,
    min_scale_bits: u64,
    max_scale_bits: u64,
}

impl MellinWgpuPlan {
    /// Create a WGPU plan descriptor carrying the Mellin scale grid.
    #[must_use]
    pub const fn new(samples: usize, min_scale_bits: u64, max_scale_bits: u64) -> Self {
        Self {
            samples,
            min_scale_bits,
            max_scale_bits,
        }
    }

    /// Return the logical sample count carried by this descriptor.
    #[must_use]
    pub const fn samples(self) -> usize {
        self.samples
    }

    /// Return the minimum Mellin scale.
    #[must_use]
    pub const fn min_scale(self) -> f64 {
        f64::from_bits(self.min_scale_bits)
    }

    /// Return the maximum Mellin scale.
    #[must_use]
    pub const fn max_scale(self) -> f64 {
        f64::from_bits(self.max_scale_bits)
    }

    /// Return whether the descriptor carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.samples == 0
    }
}
