//! WGPU plan descriptors for sliding DFT.
//!
//! The GPU direct-bins path evaluates `X[b] = sum_{n=0}^{N-1} x[n] exp(-2*pi*i*b*n/N)`
//! for b = 0..bin_count, matching `SdftPlan::direct_bins` on the CPU.

/// Metadata-preserving WGPU SDFT plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdftWgpuPlan {
    window_len: usize,
    bin_count: usize,
}

impl SdftWgpuPlan {
    /// Create a WGPU SDFT plan descriptor.
    #[must_use]
    pub const fn new(window_len: usize, bin_count: usize) -> Self {
        Self {
            window_len,
            bin_count,
        }
    }

    /// Return the sliding window length.
    #[must_use]
    pub const fn window_len(self) -> usize {
        self.window_len
    }

    /// Return the number of tracked DFT bins.
    #[must_use]
    pub const fn bin_count(self) -> usize {
        self.bin_count
    }

    /// Return the window length as the primary length accessor.
    #[must_use]
    pub const fn len(self) -> usize {
        self.window_len
    }

    /// Return whether the plan is empty (zero window or zero bins).
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.window_len == 0 || self.bin_count == 0
    }
}
