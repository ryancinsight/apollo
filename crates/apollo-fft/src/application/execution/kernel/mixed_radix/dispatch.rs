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
use super::caches::{cached_coprime_factors, cached_is_prime, cached_prime23_radices};
use super::scalar::MixedRadixScalar;

/// Authoritative single-body FFT dispatch.
///
/// `INVERSE` selects twiddle table direction and algorithm variant.
/// `NORMALIZE` gates the 1/N scale pass, eliminated at compile time when false.
#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline)]
pub(crate) fn dispatch_inplace<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [F::Complex],
    twiddles: Option<&[F::Complex]>,
) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if try_power_of_two_fast_path::<F, INVERSE, NORMALIZE>(data, twiddles) {
        return;
    }
    if F::short_winograd::<INVERSE, NORMALIZE>(data) {
        return;
    }

    let n = data.len();
    let coprime_factors = cached_coprime_factors(n);
    if let Some((n1, n2)) = coprime_factors {
        if crate::application::execution::kernel::components::good_thomas::has_static_coprime_codelet(n1, n2) {
            crate::application::execution::kernel::components::good_thomas::pfa_fft::<F>(data, INVERSE, n1, n2);
            if INVERSE && NORMALIZE {
                F::normalize(data, n);
            }
            return;
        }
    }

    if let Some(radices) = cached_prime23_radices(n) {
        match (INVERSE, NORMALIZE) {
            (false, _) => F::composite_forward(data, &radices),
            (true, false) => F::composite_inverse_unnorm(data, &radices),
            (true, true) => F::composite_inverse(data, &radices),
        }
        return;
    }

    if let Some((n1, n2)) = coprime_factors {
        crate::application::execution::kernel::components::good_thomas::pfa_fft::<F>(
            data, INVERSE, n1, n2,
        );
        if INVERSE && NORMALIZE {
            F::normalize(data, n);
        }
        return;
    }

    if cached_is_prime(n) {
        crate::application::execution::kernel::components::rader::rader_fft::<F, INVERSE>(data);
        if INVERSE && NORMALIZE {
            F::normalize(data, n);
        }
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline)]
fn try_power_of_two_fast_path<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [F::Complex],
    twiddles: Option<&[F::Complex]>,
) -> bool {
    let n = data.len();
    if !n.is_power_of_two() || n < 64 {
        return false;
    }

    if n >= crate::application::execution::kernel::tuning::FOUR_STEP_THRESHOLD {
        let use_four_step = n.trailing_zeros() % 2 == 0;
        if use_four_step {
            crate::application::execution::kernel::components::four_step::four_step_fft::<F>(
                data, INVERSE,
            );
            if INVERSE && NORMALIZE {
                F::normalize(data, n);
            }
            return true;
        }
    }

    let owned_tw;
    let tw: &[F::Complex] = match twiddles {
        Some(tw) => tw,
        None => {
            owned_tw = if INVERSE {
                F::cached_twiddle_inv(n)
            } else {
                F::cached_twiddle_fwd(n)
            };
            owned_tw.as_ref()
        }
    };
    <F as MixedRadixScalar>::with_scratch(n, |scratch| {
        if INVERSE && NORMALIZE {
            F::stockham_forward_normalized(data, scratch, tw, n);
        } else {
            F::stockham_forward(data, scratch, tw);
        }
    });
    true
}

// ── Forward ───────────────────────────────────────────────────────────────────

/// In-place forward FFT, unnormalized, for any `MixedRadixScalar` precision.
#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline)]
pub(crate) fn forward_inplace<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
) {
    dispatch_inplace::<F, false, false>(data, None);
}

// ── Inverse (unnormalized) ────────────────────────────────────────────────────

/// In-place inverse FFT, unnormalized (no 1/N division).
#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline)]
pub(crate) fn inverse_inplace_unnorm<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
) {
    dispatch_inplace::<F, true, false>(data, None);
}

// ── Inverse (normalized 1/N) ──────────────────────────────────────────────────

/// In-place inverse FFT, normalized by 1/N.
#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline)]
pub(crate) fn inverse_inplace<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
) {
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
    run_via_complex32(data, |buf| {
        dispatch_inplace::<f32, INVERSE, NORMALIZE>(buf, None);
    });
}
