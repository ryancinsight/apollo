//! Shared butterfly and small-DFT codelets (deep vertical hierarchy).
//!
//! This module collects zero-cost (monomorphized) complex butterflies, small
//! DFTs, and pointwise helpers used across Rader (convolution), Winograd,
//! Good-Thomas, Stockham, and radix-composite. All items are `pub(crate)` and
//! generic over `MixedRadixScalar` so that call sites monomorphize with zero
//! abstraction cost and no `dyn`.
//!
//! Population:
//! - `mul_conj` — conjugate multiply (Nussbaumer negacyclic CRT in Rader).
//! - `gather_unroll8` — 8-way perm gather for larger factors in md-worst GT.
//! - `dft` submodule: canonical small DFT-N codelets (dft2/3/4/5/7/8 + array
//!   variants with fused normalize) AND re-exports of composite DFTs
//!   (dft6/9/10/... up to dft484) from winograd/composite/. Single source
//!   for all small-DFT lookups — winograd, GT cook_toom, rader pairs,
//!   radix-composite, ShortDft dispatch.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

pub(crate) mod dft;

// Re-export the most-used small DFT entry points.
pub(crate) use dft::{dft2_impl, dft3_impl, dft4_array_impl, dft5_array_impl, dft7_impl};

/// Conjugate multiply: (a.re + i a.im) * (b.re - i b.im)
#[inline]
pub(crate) fn mul_conj<F: MixedRadixScalar<Complex = eunomia::Complex<F>>>(
    value: F::Complex,
    twiddle: F::Complex,
) -> F::Complex {
    eunomia::Complex::new(
        value.re * twiddle.re + value.im * twiddle.im,
        value.im * twiddle.re - value.re * twiddle.im,
    )
}

/// 8-way unrolled gather (extension of unroll4) for better ILP on larger factors in GT PFA rows
/// (and rader). Additive zero-cost; same perm loads, just wider unroll to reduce loop overhead
/// and expose more parallelism for md-worst GT sizes (198/90/84/106+ etc. that use PFA gather).
#[inline]
pub(crate) fn gather_unroll8<T: Copy>(src: &[T], perm: &[usize], dst: &mut [T]) {
    debug_assert!(dst.len() >= perm.len());
    let len = perm.len();
    let len8 = (len / 8) * 8;
    let mut q = 0usize;
    while q < len8 {
        unsafe {
            *dst.get_unchecked_mut(q) = *src.get_unchecked(*perm.get_unchecked(q));
            *dst.get_unchecked_mut(q + 1) = *src.get_unchecked(*perm.get_unchecked(q + 1));
            *dst.get_unchecked_mut(q + 2) = *src.get_unchecked(*perm.get_unchecked(q + 2));
            *dst.get_unchecked_mut(q + 3) = *src.get_unchecked(*perm.get_unchecked(q + 3));
            *dst.get_unchecked_mut(q + 4) = *src.get_unchecked(*perm.get_unchecked(q + 4));
            *dst.get_unchecked_mut(q + 5) = *src.get_unchecked(*perm.get_unchecked(q + 5));
            *dst.get_unchecked_mut(q + 6) = *src.get_unchecked(*perm.get_unchecked(q + 6));
            *dst.get_unchecked_mut(q + 7) = *src.get_unchecked(*perm.get_unchecked(q + 7));
        }
        q += 8;
    }
    while q < len {
        unsafe {
            *dst.get_unchecked_mut(q) = *src.get_unchecked(*perm.get_unchecked(q));
        }
        q += 1;
    }
}
