//! WGPU plan descriptors.

/// Metadata-preserving WGPU plan descriptor carrying both length and fractional order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrftWgpuPlan {
    len: usize,
    /// Fractional order stored as f32 bit pattern to allow Copy + Eq.
    order_bits: u32,
}

impl FrftWgpuPlan {
    /// Create a WGPU plan descriptor for a positive logical length and fractional order.
    #[must_use]
    pub const fn new(len: usize, order: f32) -> Self {
        Self {
            len,
            order_bits: order.to_bits(),
        }
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

    /// Return the fractional order carried by this descriptor.
    #[must_use]
    pub const fn order(self) -> f32 {
        f32::from_bits(self.order_bits)
    }
}

/// Metadata-preserving WGPU plan descriptor for the unitary eigendecomposition FrFT.
///
/// The unitary DFrFT is DFrFT_a(x) = V · diag(exp(−i·a·k·π/2)) · V^T · x where V is
/// the real orthogonal Grünbaum eigenvector matrix. This descriptor carries the signal
/// length and the fractional order needed to configure the three-pass GPU kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitaryFrftWgpuPlan {
    len: usize,
    /// Fractional order stored as f32 bit pattern to allow Copy + Eq.
    order_bits: u32,
}

impl UnitaryFrftWgpuPlan {
    /// Create a unitary WGPU plan descriptor for the given signal length and fractional order.
    #[must_use]
    pub const fn new(len: usize, order: f32) -> Self {
        Self {
            len,
            order_bits: order.to_bits(),
        }
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

    /// Return the fractional order carried by this descriptor.
    #[must_use]
    pub const fn order(self) -> f32 {
        f32::from_bits(self.order_bits)
    }
}
