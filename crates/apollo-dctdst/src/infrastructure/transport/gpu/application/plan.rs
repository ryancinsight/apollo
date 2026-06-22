//! WGPU plan descriptors.

use crate::RealTransformKind;

/// Metadata-preserving WGPU plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DctDstWgpuPlan {
    len: usize,
    kind: RealTransformKind,
}

impl DctDstWgpuPlan {
    /// Create a WGPU plan descriptor for a positive logical length.
    #[must_use]
    pub const fn new(len: usize, kind: RealTransformKind) -> Self {
        Self { len, kind }
    }

    /// Return the logical transform length carried by this descriptor.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return the requested transform kind.
    #[must_use]
    pub const fn kind(self) -> RealTransformKind {
        self.kind
    }

    /// Return whether the descriptor carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}
