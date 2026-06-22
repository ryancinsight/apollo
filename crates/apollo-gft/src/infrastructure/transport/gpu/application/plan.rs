//! WGPU plan descriptors.

/// Metadata-only WGPU plan descriptor carrying the graph order.
/// The eigenvector basis is supplied separately at execute time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GftWgpuPlan {
    len: usize,
}

impl GftWgpuPlan {
    /// Create a WGPU plan descriptor for a given graph order.
    #[must_use]
    pub const fn new(len: usize) -> Self {
        Self { len }
    }

    /// Return the logical graph order carried by this descriptor.
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
