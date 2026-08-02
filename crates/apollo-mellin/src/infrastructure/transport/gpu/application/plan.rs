//! WGPU plan descriptors.

/// Metadata-preserving concrete-`f32` accelerator plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MellinWgpuPlan {
    samples: usize,
    min_scale: f32,
    max_scale: f32,
}

impl MellinWgpuPlan {
    /// Create an accelerator plan descriptor carrying the concrete-`f32` scale grid.
    #[must_use]
    pub const fn new(samples: usize, min_scale: f32, max_scale: f32) -> Self {
        Self {
            samples,
            min_scale,
            max_scale,
        }
    }

    /// Return the logical sample count carried by this descriptor.
    #[must_use]
    pub const fn samples(self) -> usize {
        self.samples
    }

    /// Return the minimum Mellin scale.
    #[must_use]
    pub const fn min_scale(self) -> f32 {
        self.min_scale
    }

    /// Return the maximum Mellin scale.
    #[must_use]
    pub const fn max_scale(self) -> f32 {
        self.max_scale
    }

    /// Return whether the descriptor carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.samples == 0
    }
}
