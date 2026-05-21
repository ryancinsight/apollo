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
//! | `forward_inplace_with_twiddles`  | false   | false     |
//! | `inverse_inplace_unnorm`         | true    | false     |
//! | `inverse_inplace_unnorm_with_twiddles` | true | false  |
//! | `inverse_inplace`                | true    | true      |
//! | `inverse_inplace_with_twiddles`  | true    | true      |

use super::super::radix_shape::should_use_bluestein_instead_of_composite;
use super::caches::cached_composite_radices;
use super::scalar::MixedRadixScalar;

/// Authoritative single-body FFT dispatch.
///
/// `INVERSE` selects twiddle table direction and algorithm variant.
/// `NORMALIZE` gates the 1/N scale pass, eliminated at compile time when false.
#[inline]
fn dispatch_inplace<F: MixedRadixScalar, const INVERSE: bool, const NORMALIZE: bool>(
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
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                match (INVERSE, NORMALIZE) {
                    (false, _) => F::composite_forward(data, &radices),
                    (true, false) => F::composite_inverse_unnorm(data, &radices),
                    (true, true) => F::composite_inverse(data, &radices),
                }
                return;
            }
        }
        match (INVERSE, NORMALIZE) {
            (false, _) => F::bluestein_forward(data),
            (true, false) => F::bluestein_inverse_unnorm(data),
            (true, true) => F::bluestein_inverse(data),
        }
    }
}

// ── Forward ───────────────────────────────────────────────────────────────────

/// In-place forward FFT, unnormalized, for any `MixedRadixScalar` precision.
#[inline]
pub(crate) fn forward_inplace<F: MixedRadixScalar>(data: &mut [F::Complex]) {
    dispatch_inplace::<F, false, false>(data, None);
}

// ── Inverse (unnormalized) ────────────────────────────────────────────────────

/// In-place inverse FFT, unnormalized (no 1/N division).
#[inline]
pub(crate) fn inverse_inplace_unnorm<F: MixedRadixScalar>(data: &mut [F::Complex]) {
    dispatch_inplace::<F, true, false>(data, None);
}

// ── Inverse (normalized 1/N) ──────────────────────────────────────────────────

/// In-place inverse FFT, normalized by 1/N.
#[inline]
pub(crate) fn inverse_inplace<F: MixedRadixScalar>(data: &mut [F::Complex]) {
    dispatch_inplace::<F, true, true>(data, None);
}

// ── Backward-compatible concrete aliases ──────────────────────────────────────
//
// These thin wrappers preserve old concrete call sites in:
//   - bluestein/plan.rs   (`*_with_twiddles` variants)
//   - radix2.rs           (`*_with_twiddles` variants)
//   - dimension_1d/precision.rs (`forward_inplace_32_with_twiddles` etc.)
//
// Zero overhead: monomorphized identically to direct `dispatch_inplace::<f64/f32, ..>`.

/// In-place forward FFT (f64, unnormalized) with optional pre-computed twiddles.
#[inline]
pub fn forward_inplace_64_with_twiddles(
    data: &mut [num_complex::Complex64],
    twiddles: Option<&[num_complex::Complex64]>,
) {
    dispatch_inplace::<f64, false, false>(data, twiddles);
}
/// In-place inverse FFT (f64, normalized 1/N) with optional pre-computed twiddles.
#[inline]
pub fn inverse_inplace_64_with_twiddles(
    data: &mut [num_complex::Complex64],
    twiddles: Option<&[num_complex::Complex64]>,
) {
    dispatch_inplace::<f64, true, true>(data, twiddles);
}
/// In-place inverse FFT (f64, unnormalized) with optional pre-computed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(
    data: &mut [num_complex::Complex64],
    twiddles: Option<&[num_complex::Complex64]>,
) {
    dispatch_inplace::<f64, true, false>(data, twiddles);
}
/// In-place forward FFT (f32, unnormalized) with optional pre-computed twiddles.
#[inline]
pub fn forward_inplace_32_with_twiddles(
    data: &mut [num_complex::Complex32],
    twiddles: Option<&[num_complex::Complex32]>,
) {
    dispatch_inplace::<f32, false, false>(data, twiddles);
}
/// In-place inverse FFT (f32, normalized 1/N) with optional pre-computed twiddles.
#[inline]
pub fn inverse_inplace_32_with_twiddles(
    data: &mut [num_complex::Complex32],
    twiddles: Option<&[num_complex::Complex32]>,
) {
    dispatch_inplace::<f32, true, true>(data, twiddles);
}
/// In-place inverse FFT (f32, unnormalized) with optional pre-computed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(
    data: &mut [num_complex::Complex32],
    twiddles: Option<&[num_complex::Complex32]>,
) {
    dispatch_inplace::<f32, true, false>(data, twiddles);
}
