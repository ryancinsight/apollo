//! WGPU plan descriptors for the Haar DWT.

/// Metadata-preserving WGPU plan descriptor for the Haar Discrete Wavelet Transform.
///
/// GPU implementation supports Haar wavelet only. Input length must be a power
/// of two and satisfy `len >= 2^levels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveletWgpuPlan {
    len: usize,
    levels: usize,
}

impl WaveletWgpuPlan {
    /// Create a Haar DWT plan descriptor.
    ///
    /// Validation (power-of-two `len`, `levels > 0`, `2^levels <= len`) is
    /// enforced at the execution boundary in `WaveletWgpuBackend`.
    #[must_use]
    pub const fn new(len: usize, levels: usize) -> Self {
        Self { len, levels }
    }

    /// Return the signal length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return the number of decomposition levels.
    #[must_use]
    pub const fn levels(self) -> usize {
        self.levels
    }

    /// Return true when `len == 0` or `levels == 0`.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0 || self.levels == 0
    }
}
