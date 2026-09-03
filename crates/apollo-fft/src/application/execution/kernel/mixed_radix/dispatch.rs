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

use super::super::precision_bridge::Complex32Bridge;
use super::caches::{cached_coprime_factors, cached_is_prime, cached_prime23_radices};
use super::scalar::MixedRadixScalar;
use crate::application::execution::kernel::pot::StockhamAutosort;
use crate::with_pot_zst;

#[inline]
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
        511 => Some((73, 7)),
        _ => None,
    }
}

#[inline]
fn static_is_prime(n: usize) -> bool {
    match n {
        2 | 3 | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 | 41 | 43 | 47 | 53 | 59 | 61
        | 10007 => true,
        _ => false,
    }
}

/// Static composite radices for commonly-benchmarked sizes with all prime factors ≤ 23.
/// Avoids the runtime factoring + cache lookup overhead of `cached_prime23_radices`.
/// Radix-2 pairs are already lowered to radix-4 for optimal stage pairing.
#[inline]
pub(crate) fn static_prime23_radices(n: usize) -> Option<&'static [usize]> {
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
        72 => Some(&[4, 2, 3, 3]), // 2³×3² = 8×9 = 72
        81 => Some(&[3, 3, 3, 3]), // 3⁴
        90 => Some(&[2, 3, 3, 5]),
        96 => Some(&[4, 4, 2, 3]),     // 2⁵×3 = 32×3 = 96
        108 => Some(&[4, 3, 3, 3]),    // 2²×3³
        120 => Some(&[4, 2, 3, 5]),    // 2³×3×5
        128 => Some(&[4, 4, 4, 2]),    // 2⁷
        144 => Some(&[4, 4, 3, 3]),    // 2⁴×3² = 16×9 = 144
        162 => Some(&[2, 3, 3, 3, 3]), // 2×3⁴
        180 => Some(&[4, 3, 3, 5]),    // 2²×3²×5
        192 => Some(&[4, 4, 4, 3]),    // 2⁶×3 = 64×3 = 192
        198 => Some(&[2, 3, 3, 11]),
        216 => Some(&[4, 2, 3, 3, 3]),    // 2³×3³
        243 => Some(&[3, 3, 3, 3, 3]),    // 3⁵
        252 => Some(&[4, 3, 3, 7]),       // 2²×3²×7
        256 => Some(&[4, 4, 4, 4]),       // 2⁸
        270 => Some(&[2, 3, 3, 3, 5]),    // 2×3³×5
        288 => Some(&[4, 4, 2, 3, 3]),    // 2⁵×3² = 32×9 = 288
        324 => Some(&[4, 3, 3, 3, 3]),    // 2²×3⁴
        360 => Some(&[4, 2, 3, 3, 5]),    // 2³×3²×5
        384 => Some(&[4, 4, 4, 2, 3]),    // 2⁷×3 = 128×3 = 384
        432 => Some(&[4, 4, 3, 3, 3]),    // 2⁴×3³
        480 => Some(&[4, 4, 2, 3, 5]),    // 2⁵×3×5 = 32×15 = 480
        504 => Some(&[4, 2, 3, 3, 7]),    // 2³×3²×7
        512 => Some(&[4, 4, 4, 4, 2]),    // 2⁹
        576 => Some(&[4, 4, 4, 3, 3]),    // 2⁶×3²
        648 => Some(&[4, 2, 3, 3, 3, 3]), // 2³×3⁴
        720 => Some(&[4, 4, 3, 3, 5]),    // 2⁴×3²×5
        768 => Some(&[4, 4, 4, 4, 3]),    // 2⁸×3
        864 => Some(&[4, 4, 2, 3, 3, 3]), // 2⁵×3³ = 32×27 = 864
        960 => Some(&[4, 4, 4, 3, 5]),    // 2⁶×3×5
        972 => Some(&[4, 3, 3, 3, 3, 3]), // 2²×3⁵
        1008 => Some(&[4, 4, 3, 3, 7]),   // 2⁴×3²×7
        1024 => Some(&[4, 4, 4, 4, 4]),   // 2¹⁰
        // ── All prime-23-smooth composites ≤ 512, radix-2 pairs lowered to radix-4 ──
        50 => Some(&[2, 5, 5]),           // 2×5²
        52 => Some(&[4, 13]),             // 2²×13
        55 => Some(&[5, 11]),             // 5×11
        56 => Some(&[4, 2, 7]),           // 2³×7
        60 => Some(&[4, 3, 5]),           // 2²×3×5
        63 => Some(&[3, 3, 7]),           // 3²×7
        66 => Some(&[2, 3, 11]),          // 2×3×11
        70 => Some(&[2, 5, 7]),           // 2×5×7
        75 => Some(&[3, 5, 5]),           // 3×5²
        77 => Some(&[7, 11]),             // 7×11
        78 => Some(&[2, 3, 13]),          // 2×3×13
        80 => Some(&[4, 4, 5]),           // 2⁴×5
        84 => Some(&[4, 3, 7]),           // 2²×3×7
        88 => Some(&[4, 2, 11]),          // 2³×11
        91 => Some(&[7, 13]),             // 7×13
        98 => Some(&[2, 7, 7]),           // 2×7²
        99 => Some(&[3, 3, 11]),          // 3²×11
        100 => Some(&[4, 5, 5]),          // 2²×5²
        104 => Some(&[4, 2, 13]),         // 2³×13
        105 => Some(&[3, 5, 7]),          // 3×5×7
        110 => Some(&[2, 5, 11]),         // 2×5×11
        112 => Some(&[4, 4, 7]),          // 2⁴×7
        115 => Some(&[5, 23]),            // 5×23
        117 => Some(&[3, 3, 13]),         // 3²×13
        119 => Some(&[7, 17]),            // 7×17
        125 => Some(&[5, 5, 5]),          // 5³
        126 => Some(&[2, 3, 3, 7]),       // 2×3²×7
        130 => Some(&[2, 5, 13]),         // 2×5×13
        132 => Some(&[4, 3, 11]),         // 2²×3×11
        135 => Some(&[3, 3, 3, 5]),       // 3³×5
        140 => Some(&[4, 5, 7]),          // 2²×5×7
        147 => Some(&[3, 7, 7]),          // 3×7²
        150 => Some(&[2, 3, 5, 5]),       // 2×3×5²
        153 => Some(&[3, 3, 17]),         // 3²×17
        154 => Some(&[2, 7, 11]),         // 2×7×11
        156 => Some(&[4, 3, 13]),         // 2²×3×13
        160 => Some(&[4, 4, 2, 5]),       // 2⁵×5
        165 => Some(&[3, 5, 11]),         // 3×5×11
        168 => Some(&[4, 2, 3, 7]),       // 2³×3×7
        169 => Some(&[13, 13]),           // 13²
        175 => Some(&[5, 5, 7]),          // 5²×7
        176 => Some(&[4, 4, 11]),         // 2⁴×11
        182 => Some(&[2, 7, 13]),         // 2×7×13
        189 => Some(&[3, 3, 3, 7]),       // 3³×7
        195 => Some(&[3, 5, 13]),         // 3×5×13
        196 => Some(&[4, 7, 7]),          // 2²×7²
        200 => Some(&[4, 2, 5, 5]),       // 2³×5²
        204 => Some(&[4, 3, 17]),         // 2²×3×17
        207 => Some(&[3, 3, 23]),         // 3²×23
        208 => Some(&[4, 4, 13]),         // 2⁴×13
        210 => Some(&[2, 3, 5, 7]),       // 2×3×5×7
        220 => Some(&[4, 5, 11]),         // 2²×5×11
        224 => Some(&[4, 4, 2, 7]),       // 2⁵×7
        225 => Some(&[3, 3, 5, 5]),       // 3²×5²
        231 => Some(&[3, 7, 11]),         // 3×7×11
        234 => Some(&[2, 3, 3, 13]),      // 2×3²×13
        238 => Some(&[2, 7, 17]),         // 2×7×17
        240 => Some(&[4, 4, 3, 5]),       // 2⁴×3×5
        242 => Some(&[2, 11, 11]),        // 2×11²
        245 => Some(&[5, 7, 7]),          // 5×7²
        250 => Some(&[2, 5, 5, 5]),       // 2×5³
        253 => Some(&[11, 23]),           // 11×23
        255 => Some(&[3, 5, 17]),         // 3×5×17
        260 => Some(&[4, 5, 13]),         // 2²×5×13
        264 => Some(&[4, 2, 3, 11]),      // 2³×3×11
        272 => Some(&[4, 4, 17]),         // 2⁴×17
        273 => Some(&[3, 7, 13]),         // 3×7×13
        275 => Some(&[5, 5, 11]),         // 5²×11
        276 => Some(&[4, 3, 23]),         // 2²×3×23
        280 => Some(&[4, 2, 5, 7]),       // 2³×5×7
        286 => Some(&[2, 11, 13]),        // 2×11×13
        294 => Some(&[2, 3, 7, 7]),       // 2×3×7²
        297 => Some(&[3, 3, 3, 11]),      // 3³×11
        300 => Some(&[4, 3, 5, 5]),       // 2²×3×5²
        308 => Some(&[4, 7, 11]),         // 2²×7×11
        312 => Some(&[4, 2, 3, 13]),      // 2³×3×13
        315 => Some(&[3, 3, 5, 7]),       // 3²×5×7
        320 => Some(&[4, 4, 4, 5]),       // 2⁶×5
        325 => Some(&[5, 5, 13]),         // 5²×13
        330 => Some(&[2, 3, 5, 11]),      // 2×3×5×11
        336 => Some(&[4, 4, 3, 7]),       // 2⁴×3×7
        338 => Some(&[2, 13, 13]),        // 2×13²
        340 => Some(&[4, 5, 17]),         // 2²×5×17
        343 => Some(&[7, 7, 7]),          // 7³
        345 => Some(&[3, 5, 23]),         // 3×5×23
        350 => Some(&[2, 5, 5, 7]),       // 2×5²×7
        351 => Some(&[3, 3, 3, 13]),      // 3³×13
        352 => Some(&[4, 4, 2, 11]),      // 2⁵×11
        357 => Some(&[3, 7, 17]),         // 3×7×17
        363 => Some(&[3, 11, 11]),        // 3×11²
        364 => Some(&[4, 7, 13]),         // 2²×7×13
        368 => Some(&[4, 4, 23]),         // 2⁴×23
        375 => Some(&[3, 5, 5, 5]),       // 3×5³
        378 => Some(&[2, 3, 3, 3, 7]),    // 2×3³×7
        385 => Some(&[5, 7, 11]),         // 5×7×11
        390 => Some(&[2, 3, 5, 13]),      // 2×3×5×13
        392 => Some(&[4, 2, 7, 7]),       // 2³×7²
        396 => Some(&[4, 3, 3, 11]),      // 2²×3²×11
        400 => Some(&[4, 4, 5, 5]),       // 2⁴×5²
        405 => Some(&[3, 3, 3, 3, 5]),    // 3⁴×5
        408 => Some(&[4, 2, 3, 17]),      // 2³×3×17
        414 => Some(&[2, 3, 3, 23]),      // 2×3²×23
        416 => Some(&[4, 4, 2, 13]),      // 2⁵×13
        420 => Some(&[4, 3, 5, 7]),       // 2²×3×5×7
        425 => Some(&[5, 5, 17]),         // 5²×17
        429 => Some(&[3, 11, 13]),        // 3×11×13
        440 => Some(&[4, 2, 5, 11]),      // 2³×5×11
        441 => Some(&[3, 3, 7, 7]),       // 3²×7²
        448 => Some(&[4, 4, 4, 7]),       // 2⁶×7
        450 => Some(&[2, 3, 3, 5, 5]),    // 2×3²×5²
        455 => Some(&[5, 7, 13]),         // 5×7×13
        459 => Some(&[3, 3, 3, 17]),      // 3³×17
        462 => Some(&[2, 3, 7, 11]),      // 2×3×7×11
        468 => Some(&[4, 3, 3, 13]),      // 2²×3²×13
        476 => Some(&[4, 7, 17]),         // 2²×7×17
        483 => Some(&[3, 7, 23]),         // 3×7×23
        484 => Some(&[4, 11, 11]),        // 2²×11²
        486 => Some(&[2, 3, 3, 3, 3, 3]), // 2×3⁵
        490 => Some(&[2, 5, 7, 7]),       // 2×5×7²
        495 => Some(&[3, 3, 5, 11]),      // 3²×5×11
        500 => Some(&[4, 5, 5, 5]),       // 2²×5³
        510 => Some(&[2, 3, 5, 17]),      // 2×3×5×17
        _ => None,
    }
}

/// Authoritative single-body FFT dispatch.
///
/// `INVERSE` selects twiddle table direction and algorithm variant.
/// `NORMALIZE` gates the 1/N scale pass, eliminated at compile time when false.
#[inline]
pub(crate) fn dispatch_inplace<
    F: MixedRadixScalar<Complex = eunomia::Complex<F>>,
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

    if n == 200 {
        match (INVERSE, NORMALIZE) {
            (false, _) => F::composite_forward(data, F::COMPOSITE_RADICES_200),
            (true, false) => F::composite_inverse_unnorm(data, F::COMPOSITE_RADICES_200),
            (true, true) => F::composite_inverse(data, F::COMPOSITE_RADICES_200),
        }
        return;
    }

    // Prefer composite mixed-radix (CT) for 2/3/5/7-smooth even if coprime factors exist,
    // as GT PFA static impls have high overhead (perm, scratch, strided cols) for many sizes
    // that benchmark >2x slower than RustFFT. GT only for non-smooth coprimes or explicit static wins.
    // Check static table for ALL sizes first (zero-cost match), then fall back to cached Arc lookup.
    if let Some(radices) = static_prime23_radices(n) {
        match (INVERSE, NORMALIZE) {
            (false, _) => F::composite_forward(data, radices),
            (true, false) => F::composite_inverse_unnorm(data, radices),
            (true, true) => F::composite_inverse(data, radices),
        }
        return;
    }
    if let Some(radices) = cached_prime23_radices(n) {
        match (INVERSE, NORMALIZE) {
            (false, _) => F::composite_forward(data, &radices),
            (true, false) => F::composite_inverse_unnorm(data, &radices),
            (true, true) => F::composite_inverse(data, &radices),
        }
        return;
    }

    // Note: 90 and 198 handled by static_prime23_radices (and explicit in plan FftPlan1D for guarantee before short-win f32 policy).
    // 72 f32 now forced composite in plan (md 17x f32 via GT/Policy); static has [4,2,3,3]. Routing hardened per md-worst + "selection may not correct".

    let coprime_factors = static_coprime_factors(n).or_else(|| {
        if n > 64 {
            cached_coprime_factors(n)
        } else {
            None
        }
    });
    if let Some((n1, n2)) = coprime_factors {
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
        return;
    }

    // Every shaped strategy has declined: `n` is composite, not smooth over
    // the supported radices, and has no coprime split. Reaching the end of
    // this function used to mean returning with `data` untouched — the
    // identity, presented as a transform, with no diagnostic. Lengths of the
    // form `p^2` for a prime `p` outside the radix set (361, 841, 961, ...)
    // land here, and so did every composite built from them, because
    // Good-Thomas routes its row transforms back through this dispatcher.
    // Bluestein serves any length, so the terminal case is now a correct
    // transform rather than a silent no-op.
    crate::application::execution::kernel::components::bluestein::bluestein_fft::<
        F,
        INVERSE,
        NORMALIZE,
    >(data);
}

#[inline]
fn try_power_of_two_fast_path<
    F: MixedRadixScalar<Complex = eunomia::Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [F::Complex],
    twiddles: Option<&[F::Complex]>,
) -> bool {
    let n = data.len();
    if !n.is_power_of_two() || n < 16 {
        return false;
    }

    // PoT ZST wiring: construct exact SizedPoT<StockhamAutosort, LOG2> and pass to
    // pot_inplace_sized for zero-cost monomorphization. LOG2 < 4 excluded because
    // n < 16 returns false above; sizes 2/4/8 are handled by short_winograd.
    // LOG2 4-6 delegate to small_pot_inplace_sized which ignores twiddles,
    // so we pass an empty slice to avoid unnecessary twiddle cache lookups.
    //
    // try_pot_zst! reduces 7 identical match arms (4-10) to a single macro invocation.
    // Uses the shared with_pot_zst! from pot for ZST construction.
    macro_rules! try_pot_zst {
        ($log2:literal, $needs_twiddle:expr) => {
            with_pot_zst!($log2, _s, {
                if $needs_twiddle {
                    if let Some(tw) = twiddles {
                        F::pot_inplace_sized::<INVERSE, NORMALIZE, StockhamAutosort, $log2>(
                            data, tw, _s,
                        );
                    } else if INVERSE {
                        F::with_twiddle_inv(n, |tw| {
                            F::pot_inplace_sized::<INVERSE, NORMALIZE, StockhamAutosort, $log2>(
                                data, tw, _s,
                            );
                        });
                    } else {
                        F::with_twiddle_fwd(n, |tw| {
                            F::pot_inplace_sized::<INVERSE, NORMALIZE, StockhamAutosort, $log2>(
                                data, tw, _s,
                            );
                        });
                    }
                } else {
                    F::pot_inplace_sized::<INVERSE, NORMALIZE, StockhamAutosort, $log2>(
                        data,
                        &[],
                        _s,
                    );
                }
                return true;
            })
        };
    }
    let log2 = n.trailing_zeros();
    match log2 {
        4 => try_pot_zst!(4, false),
        5 => try_pot_zst!(5, false),
        6 => try_pot_zst!(6, false),
        7 => try_pot_zst!(7, true),
        8 => try_pot_zst!(8, true),
        9 => try_pot_zst!(9, true),
        10 => try_pot_zst!(10, true),
        _ => {}
    }

    if crate::application::execution::kernel::components::four_step::try_four_step::<
        F,
        INVERSE,
        NORMALIZE,
    >(
        data,
        crate::application::execution::kernel::tuning::FOUR_STEP_THRESHOLD,
    ) {
        return true;
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
#[inline]
pub(crate) fn forward_inplace<F: MixedRadixScalar<Complex = eunomia::Complex<F>>>(
    data: &mut [F::Complex],
) {
    dispatch_inplace::<F, false, false>(data, None);
}

// ── Inverse (unnormalized) ────────────────────────────────────────────────────

/// In-place inverse FFT, unnormalized (no 1/N division).
#[inline]
pub(crate) fn inverse_inplace_unnorm<F: MixedRadixScalar<Complex = eunomia::Complex<F>>>(
    data: &mut [F::Complex],
) {
    dispatch_inplace::<F, true, false>(data, None);
}

// ── Inverse (normalized 1/N) ──────────────────────────────────────────────────

/// In-place inverse FFT, normalized by 1/N.
#[inline]
pub(crate) fn inverse_inplace<F: MixedRadixScalar<Complex = eunomia::Complex<F>>>(
    data: &mut [F::Complex],
) {
    dispatch_inplace::<F, true, true>(data, None);
}

/// Runs the lengths whose whole `Complex32` transform is register-resident
/// over a stack buffer, so a half-storage transform of one of them touches no
/// pool at all.
///
/// `run_via_complex32` borrows a `Complex32` buffer from a thread-local pool,
/// which is the right shape when the compute buffer is large enough to be
/// worth reusing. At these lengths it is not: the buffer is at most a
/// kilobyte, the sized codelets keep it in registers throughout, and the pool
/// borrow, its depth bookkeeping and the growth check are a visible share of
/// a transform that costs a few nanoseconds.
///
/// Returns whether the length was handled.
#[inline]
pub(crate) fn try_register_resident_storage<
    S: Complex32Bridge,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [S],
) -> bool {
    /// One length: widen into a fixed stack buffer, run the sized codelet the
    /// f32 route selects for it, narrow back.
    macro_rules! sized {
        ($n:literal) => {{
            let mut staging = core::mem::MaybeUninit::<[eunomia::Complex32; $n]>::uninit();
            // SAFETY: the reference is used only as `widen_slice`'s destination
            // until every lane is written. That call fills
            // `min(src.len(), dst.len())` lanes and both are exactly `$n` here
            // — `data.len()` selected this arm — so the buffer is wholly
            // initialized before the codelet reads it. `Complex32` is a pair of
            // `f32`, which has no invalid bit pattern. Zeroing it first would
            // be a whole extra pass over a buffer the widening overwrites
            // (measured 1 to 7% of the transform at these lengths). Coverage
            // is enforced rather than argued: debug builds poison the staging
            // with NaN below, so a lane the widening failed to write poisons
            // the output and fails the value oracles.
            let buffer = unsafe { &mut *staging.as_mut_ptr() };
            #[cfg(debug_assertions)]
            buffer.fill(eunomia::Complex32::new(f32::NAN, f32::NAN));
            S::widen_slice(data, buffer);
            // SAFETY: `small_pot_inplace_sized` requires the slice to hold
            // exactly `N` samples, which the array type fixes.
            unsafe {
                <f32 as MixedRadixScalar>::small_pot_inplace_sized::<$n, INVERSE, NORMALIZE>(
                    buffer,
                );
            }
            S::narrow_slice(buffer, data);
            return true;
        }};
    }
    match data.len() {
        2 => sized!(2),
        4 => sized!(4),
        8 => sized!(8),
        16 => sized!(16),
        32 => sized!(32),
        // Not 64: its sized `small_pot` arm is the slow member of the f32
        // family here (measured 16x the plan's register-resident base64), so
        // running it over a stack buffer would only move that cost, not
        // remove it (ATLAS-APOLLO-STORAGE-ROUTE-MISSES-THE-PLAN-2026-09-02).
        _ => false,
    }
}

#[cfg(test)]
mod static_radix_table {
    use super::static_prime23_radices;

    /// Every static entry must factor the length it is keyed by.
    ///
    /// The table is hand-maintained, and an entry whose radices multiply to
    /// something other than its key is not a slow path — it is a wrong
    /// transform, because the composite kernel runs the stages it is given.
    /// A single arithmetic slip is invisible on inspection (`[4, 4, 4, 4, 3, 3]`
    /// reads like `2^6 * 3^2` until it is multiplied out) and reaches only the
    /// one length it is keyed by, so nothing but this check covers it.
    #[test]
    fn every_entry_multiplies_to_its_key() {
        let mut wrong = Vec::new();
        for n in 0..100_000usize {
            if let Some(radices) = static_prime23_radices(n) {
                let product: usize = radices.iter().product();
                if product != n {
                    wrong.push((n, radices, product));
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "static radix entries that do not factor their key \
             (length, radices, product): {wrong:?}"
        );
    }

    /// Every static entry must use radices the composite kernel can execute.
    ///
    /// `radix_composite::arity` dispatches a closed set and ends in
    /// `unreachable!("unsupported radix")`. `factorize_composite` never emits a
    /// radix outside that set, so the runtime factorizer cannot reach the
    /// panic — but this hand-written table bypasses the factorizer, and an
    /// entry naming an unsupported radix turns that `unreachable!` into a
    /// reachable panic. It is reached indirectly: a length whose own plan
    /// avoids the table still lands here when it appears as a Good-Thomas
    /// sub-transform.
    #[test]
    fn every_entry_uses_an_executable_radix() {
        // Mirrors the match arms in `radix_composite::arity::dispatch_stage`.
        const EXECUTABLE: [usize; 11] = [2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 23];
        let mut wrong = Vec::new();
        for n in 0..100_000usize {
            if let Some(radices) = static_prime23_radices(n) {
                if let Some(&bad) = radices.iter().find(|r| !EXECUTABLE.contains(r)) {
                    wrong.push((n, radices, bad));
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "static radix entries naming a radix the composite kernel cannot              execute (length, radices, offending radix): {wrong:?}"
        );
    }
}

#[cfg(test)]
mod composite_decomposition {
    use crate::application::execution::plan::fft::dimension_1d::strategy::PlanStrategy;

    /// The radix orders whose cost was measured, pinned so an edit cannot
    /// change them silently.
    ///
    /// `FftPlan1D` and `dispatch_inplace` read their decomposition from
    /// different tables, and the tables disagree about the *order* of the same
    /// factors. The order is worth tens of percent, and not in one direction:
    /// at n = 100 and n = 384 the dispatcher's static order was the faster one
    /// by 26 to 29%, which is why the plan's ladder now consults that table;
    /// at n = 180 the plan's own entry beats the static order by 62%
    /// (189 ns against 500). So neither table is authoritative, and each of
    /// these choices is a measurement rather than a rule
    /// (`backlog.md#atlas-apollo-plan-underselects-composite`). Changing one
    /// means re-measuring it.
    #[test]
    fn measured_orders_are_the_ones_selected() {
        for (n, expected) in [
            (100usize, &[4usize, 5, 5][..]),
            (384, &[4, 4, 4, 2, 3][..]),
            (180, &[5, 3, 3, 4][..]),
        ] {
            let plan = crate::FftPlan1D::<f32>::new(
                crate::Shape1D::new(n).expect("invariant: pinned lengths are non-zero"),
            );
            let PlanStrategy::Composite { radices } = &plan.strategy else {
                panic!("n = {n} is expected to plan as a composite decomposition");
            };
            assert_eq!(
                radices.as_ref(),
                expected,
                "n = {n} changed decomposition; re-measure before accepting it"
            );
        }
    }

    /// Which decomposition each route reads for the lengths whose two routes
    /// measured apart (`backlog.md#atlas-apollo-plan-underselects-composite`).
    ///
    /// `dispatch_inplace` reads `static_prime23_radices` first; the plan's
    /// selection ladder never consults it. This prints both sources beside the
    /// strategy the plan actually chose, which separates "the routes disagree
    /// about the decomposition" from "they agree and the plan's execution is
    /// slower".
    #[test]
    #[ignore = "diagnostic for the composite-length route divergence"]
    fn report_route_sources_for_divergent_lengths() {
        for n in [96usize, 100, 128, 384, 512, 1000] {
            let plan = crate::FftPlan1D::<f32>::new(
                crate::Shape1D::new(n).expect("invariant: diagnostic lengths are non-zero"),
            );
            let chosen = match &plan.strategy {
                PlanStrategy::Identity => "Identity".to_owned(),
                PlanStrategy::ShortWinograd => "ShortWinograd".to_owned(),
                PlanStrategy::PowerOfTwo { log2, .. } => format!("PowerOfTwo(log2={log2})"),
                PlanStrategy::GoodThomas { n1, n2 } => format!("GoodThomas({n1}x{n2})"),
                PlanStrategy::Composite { radices } => format!("Composite({radices:?})"),
                PlanStrategy::Rader => "Rader".to_owned(),
                PlanStrategy::Bluestein => "Bluestein".to_owned(),
            };
            let statik = super::static_prime23_radices(n);
            let cached =
                crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices(
                    n,
                );
            println!(
                "n={n:5} plan={chosen} dispatcher_static={statik:?} cached={:?}",
                cached.as_deref()
            );
        }
    }
}
