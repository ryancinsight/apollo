use super::{WgpuError, WgpuResult};

/// Plan payload for the short-time Fourier transform: frame and hop
/// lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FramePlan {
    frame_len: usize,
    hop_len: usize,
}

impl FramePlan {
    /// Create a frame-plan payload.
    ///
    /// Validation (`0 < hop_len <= frame_len`) runs at dispatch.
    #[must_use]
    pub const fn new(frame_len: usize, hop_len: usize) -> Self {
        Self { frame_len, hop_len }
    }

    /// Return the frame length carried by this payload.
    #[must_use]
    pub const fn frame_len(self) -> usize {
        self.frame_len
    }

    /// Return the hop length carried by this payload.
    #[must_use]
    pub const fn hop_len(self) -> usize {
        self.hop_len
    }

    /// Reject zero or inverted frame geometry.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan rejection naming the violated constraint.
    pub fn validate_geometry(self) -> WgpuResult<()> {
        if self.frame_len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: frame_len must be non-zero",
                    self.frame_len, self.hop_len
                ),
            });
        }
        if self.hop_len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: hop_len must be non-zero",
                    self.frame_len, self.hop_len
                ),
            });
        }
        if self.hop_len > self.frame_len {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: hop_len must not exceed frame_len",
                    self.frame_len, self.hop_len
                ),
            });
        }
        Ok(())
    }
}
