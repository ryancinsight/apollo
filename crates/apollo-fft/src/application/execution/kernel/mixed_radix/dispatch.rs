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

#[inline(always)]
fn static_coprime_factors(n: usize) -> Option<(usize, usize)> {
    match n {
        6 => Some((2, 3)),
        10 => Some((2, 5)),
        12 => Some((3, 4)),
        14 => Some((2, 7)),
        15 => Some((3, 5)),
        18 => Some((2, 9)),
        20 => Some((4, 5)),
        21 => Some((3, 7)),
        22 => Some((2, 11)),
        24 => Some((3, 8)),
        26 => Some((2, 13)),
        28 => Some((4, 7)),
        30 => Some((5, 6)),
        33 => Some((3, 11)),
        34 => Some((2, 17)),
        35 => Some((5, 7)),
        36 => Some((4, 9)),
        38 => Some((2, 19)),
        39 => Some((3, 13)),
        40 => Some((5, 8)),
        42 => Some((6, 7)),
        44 => Some((4, 11)),
        45 => Some((5, 9)),
        46 => Some((2, 23)),
        48 => Some((3, 16)),
        50 => Some((2, 25)),
        51 => Some((3, 17)),
        52 => Some((4, 13)),
        54 => Some((2, 27)),
        55 => Some((5, 11)),
        56 => Some((7, 8)),
        57 => Some((3, 19)),
        58 => Some((2, 29)),
        60 => Some((3, 20)),
        62 => Some((2, 31)),
        63 => Some((7, 9)),
        70 => Some((7, 10)),
        80 => Some((5, 16)),
        90 => Some((9, 10)),
        100 => Some((4, 25)),
        150 => Some((6, 25)),
        200 => Some((8, 25)),
        _ => None,
    }
}

#[inline(always)]
fn static_is_prime(n: usize) -> bool {
    match n {
        2 | 3 | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 | 41 | 43 | 47 | 53 | 59 | 61 | 10007 => true,
        _ => false,
    }
}

#[inline(always)]
fn static_prime23_radices(n: usize) -> Option<&'static [usize]> {
    match n {
        2 => Some(&[2]),
        3 => Some(&[3]),
        4 => Some(&[4]),
        6 => Some(&[2, 3]),
        8 => Some(&[4, 2]),
        9 => Some(&[3, 3]),
        12 => Some(&[4, 3]),
        16 => Some(&[4, 4]),
        18 => Some(&[2, 3, 3]),
        24 => Some(&[4, 2, 3]),
        27 => Some(&[3, 3, 3]),
        32 => Some(&[4, 4, 2]),
        36 => Some(&[4, 3, 3]),
        48 => Some(&[4, 4, 3]),
        54 => Some(&[2, 3, 3, 3]),
        64 => Some(&[4, 4, 4]),
        _ => None,
    }
}

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

    let coprime_factors = static_coprime_factors(n).or_else(|| {
        if n > 64 {
            cached_coprime_factors(n)
        } else {
            None
        }
    });
    if let Some((n1, n2)) = coprime_factors {
        if crate::application::execution::kernel::components::good_thomas::has_static_coprime_codelet(n1, n2) {
            crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, INVERSE>(data, n1, n2);
            if INVERSE && NORMALIZE {
                F::normalize(data, n);
            }
            return;
        }
    }

    if n <= 64 {
        if let Some(radices) = static_prime23_radices(n) {
            match (INVERSE, NORMALIZE) {
                (false, _) => F::composite_forward(data, radices),
                (true, false) => F::composite_inverse_unnorm(data, radices),
                (true, true) => F::composite_inverse(data, radices),
            }
            return;
        }
    } else if let Some(radices) = cached_prime23_radices(n) {
        match (INVERSE, NORMALIZE) {
            (false, _) => F::composite_forward(data, &radices),
            (true, false) => F::composite_inverse_unnorm(data, &radices),
            (true, true) => F::composite_inverse(data, &radices),
        }
        return;
    }

    if coprime_factors.is_some() {
        let (n1, n2) = coprime_factors.unwrap();
        crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, INVERSE>(
            data, n1, n2,
        );
        if INVERSE && NORMALIZE {
            F::normalize(data, n);
        }
        return;
    }

    let is_prime = if n <= 64 || n == 10007 {
        static_is_prime(n)
    } else {
        cached_is_prime(n)
    };
    if is_prime {
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

    match twiddles {
        Some(tw) => {
            <F as MixedRadixScalar>::with_scratch(n, |scratch| {
                if INVERSE && NORMALIZE {
                    F::stockham_forward_normalized(data, scratch, tw, n);
                } else {
                    F::stockham_forward(data, scratch, tw);
                }
            });
        }
        None => {
            if INVERSE {
                F::with_twiddle_inv(n, |tw| {
                    <F as MixedRadixScalar>::with_scratch(n, |scratch| {
                        if NORMALIZE {
                            F::stockham_forward_normalized(data, scratch, tw, n);
                        } else {
                            F::stockham_forward(data, scratch, tw);
                        }
                    });
                });
            } else {
                F::with_twiddle_fwd(n, |tw| {
                    <F as MixedRadixScalar>::with_scratch(n, |scratch| {
                        F::stockham_forward(data, scratch, tw);
                    });
                });
            }
        }
    }
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
