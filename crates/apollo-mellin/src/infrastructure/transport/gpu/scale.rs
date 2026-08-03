/// Plan payload for the Mellin transform: sample count and the scale
/// range of the generated spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalePlan {
    samples: usize,
    min_scale: f32,
    max_scale: f32,
}

impl ScalePlan {
    /// Create a scale-plan payload.
    ///
    /// Validation (positive ordered finite scales, accelerator range)
    /// runs at dispatch.
    #[must_use]
    pub const fn new(samples: usize, min_scale: f32, max_scale: f32) -> Self {
        Self {
            samples,
            min_scale,
            max_scale,
        }
    }

    /// Return the spectrum sample count carried by this payload.
    #[must_use]
    pub const fn samples(self) -> usize {
        self.samples
    }

    /// Return the smallest generated scale.
    #[must_use]
    pub const fn min_scale(self) -> f32 {
        self.min_scale
    }

    /// Return the largest generated scale.
    #[must_use]
    pub const fn max_scale(self) -> f32 {
        self.max_scale
    }
}
