/// Plan payload for the sparse Fourier transform: dense length and
/// retained support size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparsityPlan {
    len: usize,
    sparsity: usize,
}

impl SparsityPlan {
    /// Create a sparsity-plan payload.
    ///
    /// Validation (`0 < sparsity <= len`, accelerator range) runs at
    /// dispatch.
    #[must_use]
    pub const fn new(len: usize, sparsity: usize) -> Self {
        Self { len, sparsity }
    }

    /// Return the dense transform length carried by this payload.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return whether the payload carries zero dense length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Return the retained support size carried by this payload.
    #[must_use]
    pub const fn sparsity(self) -> usize {
        self.sparsity
    }
}
