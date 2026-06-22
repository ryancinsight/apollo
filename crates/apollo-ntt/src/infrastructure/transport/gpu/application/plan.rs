//! WGPU plan descriptors.

/// Metadata-preserving WGPU plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NttWgpuPlan {
    len: usize,
    modulus: u64,
    primitive_root: u64,
}

impl NttWgpuPlan {
    /// Create a WGPU plan descriptor for a positive logical length and modulus contract.
    #[must_use]
    pub const fn new(len: usize, modulus: u64, primitive_root: u64) -> Self {
        Self {
            len,
            modulus,
            primitive_root,
        }
    }

    /// Return the logical transform length carried by this descriptor.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return the modulus carried by this descriptor.
    #[must_use]
    pub const fn modulus(self) -> u64 {
        self.modulus
    }

    /// Return the primitive root carried by this descriptor.
    #[must_use]
    pub const fn primitive_root(self) -> u64 {
        self.primitive_root
    }

    /// Return whether the descriptor carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}
