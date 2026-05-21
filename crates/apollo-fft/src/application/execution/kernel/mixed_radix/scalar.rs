//! `MixedRadixScalar` — sealed precision trait driving generic FFT dispatch.
//!
//! Encodes all precision-specific operations (twiddle cache access, scratch
//! allocation, Stockham/composite/Bluestein kernel dispatch, and normalization)
//! as associated methods, allowing a single generic dispatch body to serve all
//! precisions without cloned algorithm bodies.

use super::super::radix_stage::{normalize_inplace_c32, normalize_inplace_c64};
use super::super::{bluestein, radix_composite, stockham};
use super::caches::{
    cached_twiddle_fwd_32, cached_twiddle_fwd_64, cached_twiddle_inv_32, cached_twiddle_inv_64,
    with_stockham_scratch_32, with_stockham_scratch_64,
};
use super::traits::{forward_short_winograd, inverse_short_winograd};
use num_complex::{Complex32, Complex64};
use std::sync::Arc;

mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
}

/// Sealed precision trait driving zero-cost generic FFT dispatch.
///
/// Every method is a thin delegating wrapper to the canonical concrete
/// implementation for the associated precision. The dispatch body
/// (`dispatch.rs`) is parameterized by `F: MixedRadixScalar` and calls only
/// these methods, eliminating all type-suffixed function clones at the call
/// site level.
///
/// # Safety invariant
///
/// Sealed: only `f64` and `f32` implement this trait. No downstream
/// implementation is valid.
pub(crate) trait MixedRadixScalar: private::Sealed + Sized + Copy + 'static {
    /// The complex element type operated on by this scalar: `Complex<Self>`.
    type Complex: Copy + Send + Sync + 'static;

    // ── Twiddle cache ────────────────────────────────────────────────────────

    fn cached_twiddle_fwd(n: usize) -> Arc<[Self::Complex]>;
    fn cached_twiddle_inv(n: usize) -> Arc<[Self::Complex]>;

    // ── Thread-local scratch ─────────────────────────────────────────────────

    /// Borrow the thread-local Stockham ping-pong scratch buffer of length `n`,
    /// growing it without zero-init if needed, then call `f`.
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;

    // ── Kernel dispatch ──────────────────────────────────────────────────────

    /// In-place Stockham forward pass with pre-computed twiddles.
    fn stockham_forward(
        data: &mut [Self::Complex],
        scratch: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    );

    /// Short-N Winograd DFT dispatch (sizes 2/4/8/16/32/64).
    ///
    /// Returns `true` if the length was handled, `false` otherwise.
    fn short_winograd(data: &mut [Self::Complex], inverse: bool, normalize: bool) -> bool;

    // ── Composite (2/3/5/7-smooth non-PoT) ──────────────────────────────────

    fn composite_forward(data: &mut [Self::Complex], radices: &[usize]);
    fn composite_inverse_unnorm(data: &mut [Self::Complex], radices: &[usize]);
    fn composite_inverse(data: &mut [Self::Complex], radices: &[usize]);

    // ── Bluestein (arbitrary-length non-PoT) ────────────────────────────────

    fn bluestein_forward(data: &mut [Self::Complex]);
    fn bluestein_inverse_unnorm(data: &mut [Self::Complex]);
    fn bluestein_inverse(data: &mut [Self::Complex]);

    // ── Normalization ────────────────────────────────────────────────────────

    /// Apply 1/`n` scale to every element using the fastest available path.
    fn normalize(data: &mut [Self::Complex], n: usize);
}

// ── f64 implementation ────────────────────────────────────────────────────────

impl MixedRadixScalar for f64 {
    type Complex = Complex64;

    #[inline]
    fn cached_twiddle_fwd(n: usize) -> Arc<[Complex64]> {
        cached_twiddle_fwd_64(n)
    }
    #[inline]
    fn cached_twiddle_inv(n: usize) -> Arc<[Complex64]> {
        cached_twiddle_inv_64(n)
    }
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_stockham_scratch_64(n, f)
    }
    #[inline]
    fn stockham_forward(data: &mut [Complex64], scratch: &mut [Complex64], twiddles: &[Complex64]) {
        <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
    }
    #[inline]
    fn short_winograd(data: &mut [Complex64], inverse: bool, normalize: bool) -> bool {
        if inverse {
            inverse_short_winograd(data, normalize)
        } else {
            forward_short_winograd(data)
        }
    }
    #[inline]
    fn composite_forward(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::forward_inplace_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse_unnorm(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::inverse_inplace_unnorm_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::inverse_inplace_with_radices(data, radices);
    }
    #[inline]
    fn bluestein_forward(data: &mut [Complex64]) {
        bluestein::forward_inplace_64(data);
    }
    #[inline]
    fn bluestein_inverse_unnorm(data: &mut [Complex64]) {
        bluestein::inverse_inplace_unnorm_64(data);
    }
    #[inline]
    fn bluestein_inverse(data: &mut [Complex64]) {
        bluestein::inverse_inplace_64(data);
    }
    #[inline]
    fn normalize(data: &mut [Complex64], n: usize) {
        normalize_inplace_c64(data, 1.0 / n as f64);
    }
}

// ── f32 implementation ────────────────────────────────────────────────────────

impl MixedRadixScalar for f32 {
    type Complex = Complex32;

    #[inline]
    fn cached_twiddle_fwd(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_fwd_32(n)
    }
    #[inline]
    fn cached_twiddle_inv(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_inv_32(n)
    }
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_stockham_scratch_32(n, f)
    }
    #[inline]
    fn stockham_forward(data: &mut [Complex32], scratch: &mut [Complex32], twiddles: &[Complex32]) {
        <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
    }
    #[inline]
    fn short_winograd(data: &mut [Complex32], inverse: bool, normalize: bool) -> bool {
        if inverse {
            inverse_short_winograd(data, normalize)
        } else {
            forward_short_winograd(data)
        }
    }
    #[inline]
    fn composite_forward(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::forward_inplace_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse_unnorm(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::inverse_inplace_unnorm_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::inverse_inplace_with_radices(data, radices);
    }
    #[inline]
    fn bluestein_forward(data: &mut [Complex32]) {
        bluestein::forward_inplace_32(data);
    }
    #[inline]
    fn bluestein_inverse_unnorm(data: &mut [Complex32]) {
        bluestein::inverse_inplace_unnorm_32(data);
    }
    #[inline]
    fn bluestein_inverse(data: &mut [Complex32]) {
        bluestein::inverse_inplace_32(data);
    }
    #[inline]
    fn normalize(data: &mut [Complex32], n: usize) {
        normalize_inplace_c32(data, 1.0 / n as f32);
    }
}
