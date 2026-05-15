//! Single generic FFT dispatch body parameterized by `MixedRadixScalar`.
//!
//! All routing logic lives in one `dispatch_inplace` function. `const INVERSE`
//! and `const NORMALIZE` are compile-time booleans that drive dead-code
//! elimination: the compiler emits only the branches relevant to each
//! monomorphized instantiation.
//!
//! ## Public surface
//!
//! | Function                         | INVERSE | NORMALIZE |
//! |----------------------------------|---------|-----------|
//! | `forward_inplace`                | false   | false     |
//! | `inverse_inplace_unnorm`         | true    | false     |
//! | `inverse_inplace`                | true    | true      |

use super::super::precision_bridge::{run_via_complex32, Complex32Bridge};
use super::super::radix_stage::normalize_inplace_c32;
use super::super::{radix_composite, stockham};
use super::caches::{
    cached_composite_radices, cached_twiddle_fwd_32, cached_twiddle_inv_32,
    with_stockham_scratch_32,
};
use super::scalar::MixedRadixScalar;
use super::traits::{forward_short_winograd, inverse_short_winograd};

/// Authoritative single-body FFT dispatch.
///
/// `INVERSE` selects twiddle table direction and algorithm variant.
/// `NORMALIZE` gates the 1/N scale pass, eliminated at compile time when false.
#[inline(always)]
pub(crate) fn dispatch_inplace<F: MixedRadixScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [F::Complex],
    twiddles: Option<&[F::Complex]>,
) {
    if data.len() <= 1 {
        return;
    }
    if F::short_winograd(data, INVERSE, NORMALIZE) {
        return;
    }
    if data.len().is_power_of_two() {
        // Ownership dance: borrow `twiddles` if provided, otherwise materialise
        // from the thread-local cache and hold the Arc alive for the call.
        let owned_tw;
        let tw: &[F::Complex] = match twiddles {
            Some(tw) => tw,
            None => {
                owned_tw = if INVERSE {
                    F::cached_twiddle_inv(data.len())
                } else {
                    F::cached_twiddle_fwd(data.len())
                };
                owned_tw.as_ref()
            }
        };
        F::with_scratch(data.len(), |scratch| {
            F::stockham_forward(data, scratch, tw);
        });
        // Branch eliminated at compile time when NORMALIZE is false.
        if INVERSE && NORMALIZE {
            let n = data.len();
            F::normalize(data, n);
        }
    } else {
        if let Some(radices) = cached_composite_radices(data.len()) {
            match (INVERSE, NORMALIZE) {
                (false, _) => F::composite_forward(data, &radices),
                (true, false) => F::composite_inverse_unnorm(data, &radices),
                (true, true) => F::composite_inverse(data, &radices),
            }
            return;
        }

        if let Some((n1, n2)) =
            crate::application::execution::kernel::radix_shape::coprime_factors(data.len())
        {
            crate::application::execution::kernel::good_thomas::pfa_fft::<F>(data, INVERSE, n1, n2);
            if INVERSE && NORMALIZE {
                F::normalize(data, data.len());
            }
            return;
        }

        if crate::application::execution::kernel::radix_shape::is_prime(data.len()) {
            crate::application::execution::kernel::rader::rader_fft::<F>(data, INVERSE);
            if INVERSE && NORMALIZE {
                F::normalize(data, data.len());
            }
            return;
        }
    }
}

// ── Forward ───────────────────────────────────────────────────────────────────

/// In-place forward FFT, unnormalized, for any `MixedRadixScalar` precision.
#[inline(always)]
pub(crate) fn forward_inplace<F: MixedRadixScalar>(data: &mut [F::Complex]) {
    dispatch_inplace::<F, false, false>(data, None);
}

// ── Inverse (unnormalized) ────────────────────────────────────────────────────

/// In-place inverse FFT, unnormalized (no 1/N division).
#[inline(always)]
pub(crate) fn inverse_inplace_unnorm<F: MixedRadixScalar>(data: &mut [F::Complex]) {
    dispatch_inplace::<F, true, false>(data, None);
}

// ── Inverse (normalized 1/N) ──────────────────────────────────────────────────

/// In-place inverse FFT, normalized by 1/N.
#[inline(always)]
pub(crate) fn inverse_inplace<F: MixedRadixScalar>(data: &mut [F::Complex]) {
    dispatch_inplace::<F, true, true>(data, None);
}

// ── Compact storage ──────────────────────────────────────────────────────────

/// In-place forward FFT (unnormalized) for compact storage routed through `Complex32`.
///
/// Power-of-two sizes promote to f32, run the Stockham f32 autosort kernel
/// without bit reversal, and demote back to compact storage. Non-PoT sizes use
/// the same generic selector order through `run_via_complex32`.
#[inline]
pub(crate) fn forward_compact_storage<S: Complex32Bridge>(data: &mut [S]) {
    dispatch_compact_storage::<S, false, false>(data);
}

/// In-place inverse FFT (unnormalized) for compact storage routed through `Complex32`.
#[inline]
pub(crate) fn inverse_unnorm_compact_storage<S: Complex32Bridge>(data: &mut [S]) {
    dispatch_compact_storage::<S, true, false>(data);
}

/// In-place inverse FFT normalized by 1/N for compact storage routed through `Complex32`.
#[inline]
pub(crate) fn inverse_compact_storage<S: Complex32Bridge>(data: &mut [S]) {
    dispatch_compact_storage::<S, true, true>(data);
}

#[inline]
fn dispatch_compact_storage<S: Complex32Bridge, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [S],
) {
    if data.len() <= 1 {
        return;
    }
    let n = data.len();
    if n.is_power_of_two() {
        run_via_complex32(data, |buf| {
            if INVERSE {
                if inverse_short_winograd(buf, NORMALIZE) {
                    return;
                }
                let tw = cached_twiddle_inv_32(n);
                with_stockham_scratch_32(n, |scratch| {
                    <f32 as stockham::StockhamKernel>::forward_with_scratch(
                        buf,
                        scratch,
                        tw.as_ref(),
                    );
                });
                if NORMALIZE {
                    normalize_inplace_c32(buf, 1.0f32 / n as f32);
                }
            } else {
                if forward_short_winograd(buf) {
                    return;
                }
                let tw = cached_twiddle_fwd_32(n);
                with_stockham_scratch_32(n, |scratch| {
                    <f32 as stockham::StockhamKernel>::forward_with_scratch(
                        buf,
                        scratch,
                        tw.as_ref(),
                    );
                });
            }
        });
        return;
    }

    if let Some(radices) = cached_composite_radices(n) {
        run_via_complex32(data, |buf| match (INVERSE, NORMALIZE) {
            (false, _) => radix_composite::forward_inplace_with_radices(buf, &radices),
            (true, false) => radix_composite::inverse_inplace_unnorm_with_radices(buf, &radices),
            (true, true) => radix_composite::inverse_inplace_with_radices(buf, &radices),
        });
        return;
    }

    if let Some((n1, n2)) = crate::application::execution::kernel::radix_shape::coprime_factors(n) {
        run_via_complex32(data, |buf| {
            crate::application::execution::kernel::good_thomas::pfa_fft::<f32>(
                buf, INVERSE, n1, n2,
            );
            if INVERSE && NORMALIZE {
                normalize_inplace_c32(buf, 1.0f32 / n as f32);
            }
        });
        return;
    }

    if crate::application::execution::kernel::radix_shape::is_prime(n) {
        run_via_complex32(data, |buf| {
            crate::application::execution::kernel::rader::rader_fft::<f32>(buf, INVERSE);
            if INVERSE && NORMALIZE {
                normalize_inplace_c32(buf, 1.0f32 / n as f32);
            }
        });
    }
}
