//! WGPU plan descriptors.

/// Metadata-preserving WGPU plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SftWgpuPlan {
    len: usize,
    sparsity: usize,
}

impl SftWgpuPlan {
    /// Create a WGPU plan descriptor for a positive logical length.
    #[must_use]
    pub const fn new(len: usize, sparsity: usize) -> Self {
        Self { len, sparsity }
    }

    /// Return the logical transform length carried by this descriptor.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return the retained sparse support size.
    #[must_use]
    pub const fn sparsity(self) -> usize {
        self.sparsity
    }

    /// Return whether the descriptor carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0 || self.sparsity == 0
    }
}
