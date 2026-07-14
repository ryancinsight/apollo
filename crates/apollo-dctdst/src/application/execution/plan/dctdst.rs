//! Reusable DCT/DST plan metadata surface.

use crate::domain::contracts::error::DctDstResult;
use crate::domain::metadata::kind::{RealTransformConfig, RealTransformKind};

/// Forward transform implementations.
pub mod forward;
/// Internal conversion and validation helpers.
pub mod helpers;
/// Inverse transform implementations.
pub mod inverse;
/// Typed storage transform implementations.
pub mod typed;

#[cfg(test)]
mod tests;

pub use typed::{RealTransformGpuStorage, RealTransformStorage};

/// Reusable DCT/DST plan.
///
/// The plan owns a validated real-to-real transform length and kind.
///
/// # Theorem
///
/// The DCT-II/DCT-III pair and DST-II/DST-III pair are biorthogonal under the
/// unnormalized conventions implemented in this crate:
///
/// ```text
/// DCT-III(DCT-II(x)) = (N / 2) x
/// DST-III(DST-II(x)) = (N / 2) x
/// ```
///
/// DCT-I, DCT-IV, DST-I, and DST-IV are each self-inverse under the following scales:
///
/// ```text
/// DCT-I(DCT-I(x))   = 2(N−1) · x    (N ≥ 2)
/// DCT-IV(DCT-IV(x)) = (N/2)  · x
/// DST-I(DST-I(x))   = 2(N+1) · x
/// DST-IV(DST-IV(x)) = (N/2)  · x
/// ```
///
/// Therefore `inverse` scales by `2 / N` for all type-II/III/IV pairs and by
/// `1 / (2(N−1))` or `1 / (2(N+1))` for DCT-I and DST-I respectively.
///
/// # Proof sketch
///
/// The cosine and sine basis functions used by the type-II/type-III pairs are
/// orthogonal over the half-sample shifted grid. The cross terms vanish by
/// finite trigonometric sum identities, and the diagonal terms evaluate to
/// `N / 2` under Apollo's unnormalized convention. DCT-I and DST-I carry an
/// explicit factor of 2 in their definitions; their orthogonality diagonals
/// evaluate to `(N−1)` and `(N+1)` respectively, yielding the stated scales.
///
/// # Complexity
///
/// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
/// analytical kernel). Both paths use O(1) auxiliary storage for caller-owned
/// `*_into` paths (the fast path allocates a 2N complex buffer internally for
/// the FFT work area).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DctDstPlan {
    config: RealTransformConfig,
}

impl DctDstPlan {
    /// Create a validated DCT/DST plan.
    pub fn new(len: usize, kind: RealTransformKind) -> DctDstResult<Self> {
        Ok(Self {
            config: RealTransformConfig::new(len, kind)?,
        })
    }

    /// Return the validated configuration.
    #[must_use]
    pub const fn config(self) -> RealTransformConfig {
        self.config
    }

    /// Return transform length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.config.len()
    }

    /// Return true when transform length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.config.is_empty()
    }

    /// Return transform kind.
    #[must_use]
    pub const fn kind(self) -> RealTransformKind {
        self.config.kind()
    }
}
