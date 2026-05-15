//! `MixedRadixScalar` — sealed precision trait driving generic FFT dispatch.
//!
//! Encodes all precision-specific operations (twiddle cache access, scratch
//! allocation, Stockham/composite/Rader kernel dispatch, and normalization)
//! as associated methods, allowing a single generic dispatch body to serve all
//! precisions without cloned algorithm bodies.

use super::super::radix_stage::{normalize_inplace_c32, normalize_inplace_c64};
use super::super::{radix_composite, stockham};
use super::caches::{
    cached_twiddle_fwd_32, cached_twiddle_fwd_64, cached_twiddle_inv_32, cached_twiddle_inv_64,
    with_stockham_scratch_32, with_stockham_scratch_64,
};
use super::traits::{forward_short_winograd, inverse_short_winograd};
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

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
    /// The complex element type operated on by this scalar.
    type Complex: Copy
        + Send
        + Sync
        + 'static
        + num_traits::Zero
        + std::ops::Add<Output = Self::Complex>
        + std::ops::Mul<Output = Self::Complex>;

    /// Create a complex number from f64 real and imaginary parts.
    fn complex(re: f64, im: f64) -> Self::Complex;

    // ── Twiddle cache ────────────────────────────────────────────────────────

    fn cached_twiddle_fwd(n: usize) -> Arc<[Self::Complex]>;
    fn cached_twiddle_inv(n: usize) -> Arc<[Self::Complex]>;

    fn cached_rader_spectrum(
        n: usize,
        inverse: bool,
        generator_inverse: usize,
    ) -> Arc<[Self::Complex]>;

    // ── Thread-local scratch ─────────────────────────────────────────────────

    /// Borrow the thread-local Stockham ping-pong scratch buffer of length `n`,
    /// growing it without zero-init if needed, then call `f`.
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;

    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;
    fn with_rader_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;

    // ── Vectorized operations ────────────────────────────────────────────────

    /// Point-wise complex multiplication `a[i] = a[i] * b[i]`.
    /// `a` and `b` must have the same length.
    fn pointwise_mul(a: &mut [Self::Complex], b: &[Self::Complex]);

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

    // ── Normalization ────────────────────────────────────────────────────────

    /// Apply 1/`n` scale to every element using the fastest available path.
    fn normalize(data: &mut [Self::Complex], n: usize);
}

// ── f64 implementation ────────────────────────────────────────────────────────

impl MixedRadixScalar for f64 {
    type Complex = Complex64;

    #[inline]
    fn complex(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

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
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        super::caches::with_pfa_scratch_64(n, f)
    }
    #[inline]
    fn with_rader_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        super::caches::with_rader_scratch_64(n, f)
    }
    #[inline]
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        super::caches::with_rader_padded_scratch_64(n, f)
    }

    #[inline]
    fn cached_rader_spectrum(
        n: usize,
        inverse: bool,
        generator_inverse: usize,
    ) -> Arc<[Complex64]> {
        static CACHE: LazyLock<RwLock<HashMap<(usize, bool, usize), Arc<[Complex64]>>>> =
            LazyLock::new(|| RwLock::new(HashMap::new()));
        cached_rader_spectrum_impl::<f64, _>(&CACHE, n, inverse, generator_inverse)
    }

    #[inline]
    fn pointwise_mul(a: &mut [Complex64], b: &[Complex64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x *= *y;
        }
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
    fn normalize(data: &mut [Complex64], n: usize) {
        normalize_inplace_c64(data, 1.0 / n as f64);
    }
}

// ── f32 implementation ────────────────────────────────────────────────────────

impl MixedRadixScalar for f32 {
    type Complex = Complex32;

    #[inline]
    fn complex(re: f64, im: f64) -> Complex32 {
        Complex32::new(re as f32, im as f32)
    }

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
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::with_pfa_scratch_32(n, f)
    }
    #[inline]
    fn with_rader_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::with_rader_scratch_32(n, f)
    }
    #[inline]
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::with_rader_padded_scratch_32(
            n, f,
        )
    }

    #[inline]
    fn cached_rader_spectrum(
        n: usize,
        inverse: bool,
        generator_inverse: usize,
    ) -> Arc<[Complex32]> {
        static CACHE: LazyLock<RwLock<HashMap<(usize, bool, usize), Arc<[Complex32]>>>> =
            LazyLock::new(|| RwLock::new(HashMap::new()));
        cached_rader_spectrum_impl::<f32, _>(&CACHE, n, inverse, generator_inverse)
    }

    #[inline]
    fn pointwise_mul(a: &mut [Complex32], b: &[Complex32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x *= *y;
        }
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
    fn normalize(data: &mut [Complex32], n: usize) {
        normalize_inplace_c32(data, 1.0 / n as f32);
    }
}

#[inline]
fn cached_rader_spectrum_impl<F, T>(
    cache: &'static LazyLock<RwLock<HashMap<(usize, bool, usize), Arc<[T]>>>>,
    n: usize,
    inverse: bool,
    generator_inverse: usize,
) -> Arc<[T]>
where
    F: MixedRadixScalar<Complex = T>,
    T: Copy + Send + Sync + 'static,
{
    let key = (n, inverse, generator_inverse);
    if let Some(spectrum) = cache.read().get(&key).cloned() {
        return spectrum;
    }

    let spectrum = build_rader_spectrum::<F>(n, inverse, generator_inverse);
    cache
        .write()
        .entry(key)
        .or_insert_with(|| Arc::clone(&spectrum))
        .clone()
}

fn build_rader_spectrum<F: MixedRadixScalar>(
    n: usize,
    inverse: bool,
    generator_inverse: usize,
) -> Arc<[F::Complex]> {
    let l = n - 1;
    let padded_len = (2 * l - 1).next_power_of_two();
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut kernel = vec![F::complex(0.0, 0.0); padded_len];
    let mut curr_inv = 1;
    for value in kernel.iter_mut().take(l) {
        let angle = sign * std::f64::consts::TAU * (curr_inv as f64) / (n as f64);
        *value = F::complex(angle.cos(), angle.sin());
        curr_inv = (curr_inv * generator_inverse) % n;
    }
    crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(&mut kernel);
    Arc::from(kernel.into_boxed_slice())
}
