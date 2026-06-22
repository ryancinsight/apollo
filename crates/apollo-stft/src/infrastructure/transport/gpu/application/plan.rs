//! WGPU plan descriptors.

/// Metadata-preserving WGPU plan descriptor for the Short-Time Fourier Transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StftWgpuPlan {
    frame_len: usize,
    hop_len: usize,
}
impl StftWgpuPlan {
    /// Create a WGPU STFT plan descriptor.
    #[must_use]
    pub const fn new(frame_len: usize, hop_len: usize) -> Self {
        Self { frame_len, hop_len }
    }

    /// Return the frame length.
    #[must_use]
    pub const fn frame_len(self) -> usize {
        self.frame_len
    }

    /// Return the hop length.
    #[must_use]
    pub const fn hop_len(self) -> usize {
        self.hop_len
    }

    /// Return the frame length as the primary length accessor.
    #[must_use]
    pub const fn len(self) -> usize {
        self.frame_len
    }

    /// Return whether the plan is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.frame_len == 0 || self.hop_len == 0
    }
}
