//! WGPU plan descriptors.

/// Metadata-preserving WGPU plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HilbertWgpuPlan {
    len: usize,
}

impl HilbertWgpuPlan {
    /// Create a WGPU plan descriptor for a positive logical length.
    #[must_use]
    pub const fn new(len: usize) -> Self {
        Self { len }
    }

    /// Return the logical transform length carried by this descriptor.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return whether the descriptor carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}
