#![warn(missing_docs)]
//! WGPU backend boundary for Apollo FrFT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037). This crate ships two fractional-Fourier algorithms as two
//! executor markers over one order-carrying payload: the direct sampled
//! DFrFT ([`FrftGpuKernel`]) and the Candan--Grunbaum unitary DFrFT
//! ([`UnitaryFrftGpuKernel`]), each with its own typed plan and backend.

/// Infrastructure boundary for the FrFT kernels.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::FrftGpuKernel;
pub use infrastructure::unitary_kernel::UnitaryFrftGpuKernel;

/// Plan payload for a fractional-order transform: logical length and the
/// fractional order stored as an IEEE bit pattern so the payload stays
/// `Eq`-clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderPlan {
    len: usize,
    order_bits: u32,
}

impl OrderPlan {
    /// Create an order-plan payload for a logical length and fractional
    /// order.
    #[must_use]
    pub const fn new(len: usize, order: f32) -> Self {
        Self {
            len,
            order_bits: order.to_bits(),
        }
    }

    /// Return the logical transform length carried by this payload.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return whether the payload carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Return the fractional order carried by this payload.
    #[must_use]
    pub const fn order(self) -> f32 {
        f32::from_bits(self.order_bits)
    }
}

/// Metadata-preserving WGPU plan descriptor for the direct sampled DFrFT.
pub type FrftWgpuPlan = apollo_fft::WgpuTransformPlan<FrftGpuKernel>;

/// WGPU backend descriptor for the direct sampled DFrFT.
pub type FrftWgpuBackend = apollo_fft::WgpuTransformBackend<FrftGpuKernel>;

/// Metadata-preserving WGPU plan descriptor for the unitary DFrFT.
pub type UnitaryFrftWgpuPlan = apollo_fft::WgpuTransformPlan<UnitaryFrftGpuKernel>;

/// WGPU backend descriptor for the unitary DFrFT.
pub type UnitaryFrftWgpuBackend = apollo_fft::WgpuTransformBackend<UnitaryFrftGpuKernel>;
