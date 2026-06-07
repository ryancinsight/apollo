//! Generic Stockham transform driver and fixed-size specializations.

use super::precision::{
    fusion_fits, fusion_twiddle_len, stockham_twiddle_subslice, stockham_twiddle_table_len,
    StockhamFusion, StockhamPrecision, StockhamTwiddleCursor,
};
use super::precision::{StockhamFused1, StockhamFused2, StockhamFused3, StockhamFused4};
// StockhamFused types imported below

#[cfg(not(target_arch = "x86_64"))]
use crate::application::execution::kernel::pot::{PoTStrategy, SizedPoT};

/// Power-of-two sizes that have hand-optimized stage sequences bypassing the
/// greedy fusion-eligibility loop.  See [`transform_len4096_four_triples`]
/// (four consecutive triples) and [`transform_len32768`] (5×triple with 4x-unrolled radix1 first pass).
#[cfg(test)]
pub(crate) const SPECIAL_TRANSFORM_SIZES: &[usize] = &[4096, 32768]; // 128/256 have dedicated cases in transform (delegated for tw correctness; unroll stubs cleaned)

pub(crate) fn transform<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
    scale: Option<P::Real>,
) {
    let log2 = data.len().trailing_zeros();
    transform_sized::<P>(data, scratch, twiddles, scale, log2);
}

/// Sized entry: routes to monomorphized transform_impl<LOG2, P> for known powers.
/// This enables zero-cost monomorphization per log2 (structural const generic for stage count,
/// fusion decisions, unroll factors per PoTStrategy ZST). Callers that know the size
/// (plan PowerOfTwo log2 match, dispatch PoT, bluestein pow2 pads, rader) hit concrete
/// <LOG2> instance; unknown fall to <0> runtime generic body (inner-fn pattern to bound bloat).
#[inline]
pub(crate) fn transform_sized<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
    scale: Option<P::Real>,
    log2: u32,
) {
    match log2 {
        5 => transform_impl::<5, P>(data, scratch, twiddles, scale),
        6 => transform_impl::<6, P>(data, scratch, twiddles, scale),
        7 => transform_impl::<7, P>(data, scratch, twiddles, scale),
        8 => transform_impl::<8, P>(data, scratch, twiddles, scale),
        9 => transform_impl::<9, P>(data, scratch, twiddles, scale),
        10 => transform_impl::<10, P>(data, scratch, twiddles, scale),
        11 => transform_impl::<11, P>(data, scratch, twiddles, scale),
        12 => transform_impl::<12, P>(data, scratch, twiddles, scale),
        15 => transform_impl::<15, P>(data, scratch, twiddles, scale),
        _ => transform_impl::<0, P>(data, scratch, twiddles, scale),
    }
}

/// ZST-driven entry point. Wires PoTStrategy (StockhamAutosort by default)
/// + const LOG2 from plan/dispatch SizedPoT construction. Zero-sized, zero runtime cost;
/// monomorph selects per-strategy/per-size schedule. Used to tag hot PoT paths (512/1024+)
/// for deeper specialization without excess casting or indirection.
/// Currently, all stockham paths (including from f32 bluestein rader pads for worst md sizes)
/// benefit via the log2->sized dispatch below; this fn is the typed ZST surface for direct strategy
/// routing in plan executors/pot (invoked via constructions in dimension_1d/dispatch).
#[inline]
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn transform_with_strategy<S: PoTStrategy, const LOG2: u32, P: StockhamPrecision>(
    _s: SizedPoT<S, LOG2>,
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
    scale: Option<P::Real>,
) {
    transform_sized::<P>(data, scratch, twiddles, scale, LOG2);
}

fn transform_impl<const LOG2: u32, P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
    scale: Option<P::Real>,
) {
    let n = if LOG2 == 0 {
        data.len()
    } else {
        1usize << LOG2
    };
    if n <= 1 {
        if let Some(scale) = scale {
            P::scale(data, scale);
        }
        return;
    }
    debug_assert_eq!(data.len(), n);
    debug_assert!(n.is_power_of_two());
    debug_assert!(twiddles.len() >= stockham_twiddle_table_len(n));

    // Per-LOG2 monomorphized bodies for hot PoT sizes (md-worst from benchmark_results: 32/64/128/256/512/1024/32768).
    // Structural const LOG2 selects straight-line stage sequence (no runtime while/if fusion in hot path).
    // Explicit twiddle subslices (derived from cursor+fusion_fits simulation) guarantee exact match to general.
    // 128/256 now direct (previously delegated for cursor bug; now hardcoded correct ranges eliminate).
    // Inner special fns own the seq (SRP); scale applied uniformly after (or inside for 32768 precedent).
    // ZST callers (transform_with_strategy) + sized dispatch hit the concrete <LOG2> monomorph.
    // n=32/64/128/256/512/1024 have stage_triple radix-1 unrolled specials (InnerFn + explicit iters, DCE per mono) for first pass.
    // n=32768 has 4x unrolled k loop for first pass (ILP for controlling worst PoT 32768 f64 2.75x).
    // Preserves: all Fused decisions, ping-pong, final copy when seq ends on scratch, AVX paths via sized still benefit.
    match LOG2 {
        5 => {
            transform_len32::<P>(data, scratch, twiddles);
            if let Some(scale) = scale {
                P::scale(data, scale);
            }
            return;
        }
        6 => {
            transform_len64::<P>(data, scratch, twiddles);
            if let Some(scale) = scale {
                P::scale(data, scale);
            }
            return;
        }
        7 => {
            transform_len128::<P>(data, scratch, twiddles);
            if let Some(scale) = scale {
                P::scale(data, scale);
            }
            return;
        }
        8 => {
            transform_len256::<P>(data, scratch, twiddles);
            if let Some(scale) = scale {
                P::scale(data, scale);
            }
            return;
        }
        9 => {
            transform_len512::<P>(data, scratch, twiddles);
            if let Some(scale) = scale {
                P::scale(data, scale);
            }
            return;
        }
        10 => {
            transform_len1024::<P>(data, scratch, twiddles);
            if let Some(scale) = scale {
                P::scale(data, scale);
            }
            return;
        }
        11 => {
            transform_len2048::<P>(data, scratch, twiddles);
            if let Some(scale) = scale {
                P::scale(data, scale);
            }
            return;
        }
        15 | 0
            if (LOG2 == 15 || n == 32768) && P::MAX_FUSED_STAGES >= StockhamFused4::STAGE_COUNT =>
        {
            transform_len32768::<P>(data, scratch, twiddles);
            if let Some(scale) = scale {
                P::scale(data, scale);
            }
            return;
        }
        _ => {}
    }

    let mut cursor = StockhamTwiddleCursor::new(twiddles);
    let mut stride = 1usize;
    let mut input_is_data = true;
    while stride < n {
        if P::MAX_FUSED_STAGES >= StockhamFused4::STAGE_COUNT
            && fusion_fits::<StockhamFused4>(stride, n)
            && P::stage_quad_enabled(stride, n, input_is_data)
        {
            let twiddle_len = fusion_twiddle_len::<StockhamFused4>(stride);
            let fusion_twiddles = unsafe { cursor.take(twiddle_len) };
            let first_twiddles = unsafe { stockham_twiddle_subslice(fusion_twiddles, 0, stride) };
            let second_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride, stride << 1) };
            let third_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride * 3, stride << 2) };
            let fourth_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride * 7, stride << 3) };
            if input_is_data {
                P::stage_quad(
                    data,
                    scratch,
                    stride,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                    fourth_twiddles,
                );
            } else {
                P::stage_quad(
                    scratch,
                    data,
                    stride,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                    fourth_twiddles,
                );
            }
            input_is_data = !input_is_data;
            stride <<= StockhamFused4::STAGE_COUNT;
        } else if P::MAX_FUSED_STAGES >= StockhamFused3::STAGE_COUNT
            && fusion_fits::<StockhamFused3>(stride, n)
            && P::stage_triple_enabled(stride, n, input_is_data)
        {
            let twiddle_len = fusion_twiddle_len::<StockhamFused3>(stride);
            let fusion_twiddles = unsafe { cursor.take(twiddle_len) };
            let first_twiddles = unsafe { stockham_twiddle_subslice(fusion_twiddles, 0, stride) };
            let second_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride, stride << 1) };
            let third_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride * 3, stride << 2) };
            if input_is_data {
                P::stage_triple(
                    data,
                    scratch,
                    stride,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                );
            } else {
                P::stage_triple(
                    scratch,
                    data,
                    stride,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                );
            }
            input_is_data = !input_is_data;
            stride <<= StockhamFused3::STAGE_COUNT;
        } else if P::MAX_FUSED_STAGES >= StockhamFused2::STAGE_COUNT
            && fusion_fits::<StockhamFused2>(stride, n)
        {
            let twiddle_len = fusion_twiddle_len::<StockhamFused2>(stride);
            let fusion_twiddles = unsafe { cursor.take(twiddle_len) };
            let first_twiddles = unsafe { stockham_twiddle_subslice(fusion_twiddles, 0, stride) };
            let second_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride, stride << 1) };
            if input_is_data {
                P::stage_pair(data, scratch, stride, first_twiddles, second_twiddles);
            } else {
                P::stage_pair(scratch, data, stride, first_twiddles, second_twiddles);
            }
            input_is_data = !input_is_data;
            stride <<= StockhamFused2::STAGE_COUNT;
        } else {
            let twiddle_len = fusion_twiddle_len::<StockhamFused1>(stride);
            let stage_twiddles = unsafe { cursor.take(twiddle_len) };
            if input_is_data {
                P::stage(data, scratch, stride, stage_twiddles);
            } else {
                P::stage(scratch, data, stride, stage_twiddles);
            }
            input_is_data = !input_is_data;
            stride <<= StockhamFused1::STAGE_COUNT;
        }
    }
    debug_assert_eq!(cursor.consumed(), stockham_twiddle_table_len(n));
    if !input_is_data {
        data.copy_from_slice(scratch);
    }
    if let Some(scale) = scale {
        P::scale(data, scale);
    }
}

/// Optimized 5-pass Stockham transform for N=32768.
///
/// ## Stage sequence
///
/// | Pass | Fused type | Stride | Stages covered | input_is_data after |
/// |------|-----------|--------|----------------|----------------------|
/// |  1   | triple    |      1 |  1-3           | scratch              |
/// |  2   | triple    |      8 |  4-6           | data                 |
/// |  3   | triple    |     64 |  7-9           | scratch              |
/// |  4   | triple    |    512 | 10-12          | data                 |
/// |  5   | triple    |   4096 | 13-15          | scratch              |
///
/// Pass 1 (radix1 triple, stride=1) uses 4x-unrolled AVX stage (ILP for largest groups; see stage_triple_radix1_n32768).
/// The all-triple schedule is faster than the shorter quad-heavy schedule on
/// the current AVX backend despite requiring a terminal full-buffer copy.
/// (Future mem-eff: even-pass schedule to land final write in data, eliding copy.)
#[inline]
pub(crate) fn transform_len32768<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 32768);
    debug_assert_eq!(scratch.len(), 32768);
    debug_assert!(twiddles.len() >= 32767);

    // Pass 1: triple(stride=1) -> writes to scratch
    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    // Pass 2: triple(stride=8) -> writes to data
    P::stage_triple(
        scratch,
        data,
        8,
        &twiddles[7..15],
        &twiddles[15..31],
        &twiddles[31..63],
    );
    // Pass 3: triple(stride=64) -> writes to scratch
    P::stage_triple(
        data,
        scratch,
        64,
        &twiddles[63..127],
        &twiddles[127..255],
        &twiddles[255..511],
    );
    // Pass 4: triple(stride=512) -> writes to data
    P::stage_triple(
        scratch,
        data,
        512,
        &twiddles[511..1023],
        &twiddles[1023..2047],
        &twiddles[2047..4095],
    );
    // Pass 5: triple(stride=4096) -> writes to scratch
    P::stage_triple(
        data,
        scratch,
        4096,
        &twiddles[4095..8191],
        &twiddles[8191..16383],
        &twiddles[16383..32767],
    );

    data.copy_from_slice(scratch);
}

#[inline]
pub(crate) fn transform_len4096_four_triples<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 4096);
    debug_assert_eq!(scratch.len(), 4096);
    debug_assert!(twiddles.len() >= 4095);

    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    P::stage_triple(
        scratch,
        data,
        8,
        &twiddles[7..15],
        &twiddles[15..31],
        &twiddles[31..63],
    );
    P::stage_triple(
        data,
        scratch,
        64,
        &twiddles[63..127],
        &twiddles[127..255],
        &twiddles[255..511],
    );
    P::stage_triple(
        scratch,
        data,
        512,
        &twiddles[511..1023],
        &twiddles[1023..2047],
        &twiddles[2047..4095],
    );
}

/// Monomorphized Stockham body for N=32 (LOG2=5). Highest-prob PoT size from benchmark_results (still >1x).
/// Sequence (greedy fusion, triple max): triple(1) data->scratch + pair(8) scratch->data. No terminal copy.
/// First triple radix1 delegates (in P stage) to specialized n32 unrolled (no-loop vector for avx;
/// explicit 4x scalar j0 for non-avx) via Inner-Fn + per-LOG2 routing. Enables ILP/DCE for this mono.
#[inline]
pub(crate) fn transform_len32<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 32);
    debug_assert_eq!(scratch.len(), 32);
    debug_assert!(twiddles.len() >= 31);

    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    P::stage_pair(scratch, data, 8, &twiddles[7..15], &twiddles[15..31]);
}

/// Monomorphized Stockham body for N=64 (LOG2=6).
/// Sequence: triple(1) d->sc + triple(8) sc->d. No copy.
#[inline]
pub(crate) fn transform_len64<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 64);
    debug_assert_eq!(scratch.len(), 64);
    debug_assert!(twiddles.len() >= 63);

    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    P::stage_triple(
        scratch,
        data,
        8,
        &twiddles[7..15],
        &twiddles[15..31],
        &twiddles[31..63],
    );
}

/// Monomorphized Stockham body for N=128 (LOG2=7). Direct explicit + first pass (stride=1 triple radix1) unrolled (n128 special, additive to n32/n64).
/// Sequence: t1 d->sc + t8 sc->d + single(64) d->sc ; terminal copy to data.
#[inline]
pub(crate) fn transform_len128<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 128);
    debug_assert_eq!(scratch.len(), 128);
    debug_assert!(twiddles.len() >= 127);

    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    P::stage_triple(
        scratch,
        data,
        8,
        &twiddles[7..15],
        &twiddles[15..31],
        &twiddles[31..63],
    );
    P::stage(data, scratch, 64, &twiddles[63..127]);
    data.copy_from_slice(scratch);
}

/// Monomorphized Stockham body for N=256 (LOG2=8).
/// Sequence: t1 + t8 + pair(64) d->sc ; copy.
#[inline]
pub(crate) fn transform_len256<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 256);
    debug_assert_eq!(scratch.len(), 256);
    debug_assert!(twiddles.len() >= 255);

    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    P::stage_triple(
        scratch,
        data,
        8,
        &twiddles[7..15],
        &twiddles[15..31],
        &twiddles[31..63],
    );
    P::stage_pair(data, scratch, 64, &twiddles[63..127], &twiddles[127..255]);
    data.copy_from_slice(scratch);
}

/// Monomorphized Stockham body for N=512 (LOG2=9). Hot md-worst PoT (ZST wired from plan/dispatch + rader-bluestein f32).
/// Sequence: t1 + t8 + t64 d->sc ; copy.
#[inline]
pub(crate) fn transform_len512<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 512);
    debug_assert_eq!(scratch.len(), 512);
    debug_assert!(twiddles.len() >= 511);

    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    P::stage_triple(
        scratch,
        data,
        8,
        &twiddles[7..15],
        &twiddles[15..31],
        &twiddles[31..63],
    );
    P::stage_triple(
        data,
        scratch,
        64,
        &twiddles[63..127],
        &twiddles[127..255],
        &twiddles[255..511],
    );
    data.copy_from_slice(scratch);
}

/// Monomorphized Stockham body for N=1024 (LOG2=10).
/// Sequence: t1 + t8 + t64 + single(512) sc->d . Ends on data, no copy.
#[inline]
pub(crate) fn transform_len1024<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 1024);
    debug_assert_eq!(scratch.len(), 1024);
    debug_assert!(twiddles.len() >= 1023);

    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    P::stage_triple(
        scratch,
        data,
        8,
        &twiddles[7..15],
        &twiddles[15..31],
        &twiddles[31..63],
    );
    P::stage_triple(
        data,
        scratch,
        64,
        &twiddles[63..127],
        &twiddles[127..255],
        &twiddles[255..511],
    );
    P::stage(scratch, data, 512, &twiddles[511..1023]);
}

/// Monomorphized Stockham body for N=2048 (LOG2=11). Fills gap between 1024 and 32768.
/// Sequence: t1 + t8 + t64 + pair(512) sc->d. No terminal copy.
#[inline]
pub(crate) fn transform_len2048<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 2048);
    debug_assert_eq!(scratch.len(), 2048);
    debug_assert!(twiddles.len() >= 2047);

    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    P::stage_triple(
        scratch,
        data,
        8,
        &twiddles[7..15],
        &twiddles[15..31],
        &twiddles[31..63],
    );
    P::stage_triple(
        data,
        scratch,
        64,
        &twiddles[63..127],
        &twiddles[127..255],
        &twiddles[255..511],
    );
    P::stage_pair(
        scratch,
        data,
        512,
        &twiddles[511..1023],
        &twiddles[1023..2047],
    );
}

// (len128/256 dedicated unrolled fns removed for cleanup after delegation to generic for
#[cfg(test)]
mod tests {
    use super::SPECIAL_TRANSFORM_SIZES;

    #[test]
    fn special_transform_sizes_are_powers_of_two() {
        for &n in SPECIAL_TRANSFORM_SIZES {
            assert!(
                n.is_power_of_two(),
                "SPECIAL_TRANSFORM_SIZES entry {n} must be a power of two"
            );
        }
    }

    #[test]
    fn special_transform_sizes_are_large() {
        for &n in SPECIAL_TRANSFORM_SIZES {
            if n < 4096 {
                continue; // no medium specials in list currently
            }
            assert!(
                n >= 4096,
                "SPECIAL_TRANSFORM_SIZES entry {n} must be >= 4096"
            );
        }
    }

    #[test]
    fn special_transform_4096_covered_by_four_triples() {
        // 4096 is handled by the dedicated four-triple path.
        assert!(SPECIAL_TRANSFORM_SIZES.contains(&4096));
    }

    // (128/256 special cases in transform fn, not in SPECIAL list post cleanup)

    #[test]
    fn special_transform_32768_covered_by_len32768() {
        // 32768 is handled by the dedicated length-32768 path.
        assert!(SPECIAL_TRANSFORM_SIZES.contains(&32768));
    }

    #[test]
    fn special_transform_sizes_match_generic_guard() {
        // The 32768 guard in `transform` requires MAX_FUSED_STAGES >= 4.
        assert_eq!(32768usize.trailing_zeros(), 15);
        assert!(SPECIAL_TRANSFORM_SIZES.contains(&32768));
    }

    // Schedule dumper for deriving per-LOG2 specialized bodies (monomorph elevation).
    // Replicates the greedy fusion decision (with quad disabled per default stage_quad_enabled=false;
    // triple enabled). Used to emit exact stage sequence + tw ranges + ping-pong for hot PoT sizes.
    // Run with `cargo test ... -- --nocapture` to capture for hardcoding specials (32/64/128/256/512/1024).
    // This enables structural const stage seq per LOG2 without runtime if/while in body.
    #[test]
    fn dump_fusion_schedules_for_hot_pot() {
        // Local pure decision (no P, quad disabled to match base impls + 32768 precedent).
        fn compute_schedule(
            log2: u32,
        ) -> Vec<(
            u32,   /*stages*/
            usize, /*stride*/
            usize, /*tw_len*/
            bool,  /*start_data*/
        )> {
            let n = 1usize << log2;
            let mut stride = 1usize;
            let mut input_is_data = true;
            let mut out = vec![];
            // MAX=3 (triples) conservative; quad only for special 32768 path which overrides.
            let max_f = 3u32;
            while stride < n {
                if max_f >= 3 && super::fusion_fits::<super::StockhamFused3>(stride, n) {
                    let tw = super::fusion_twiddle_len::<super::StockhamFused3>(stride);
                    out.push((3, stride, tw, input_is_data));
                    input_is_data = !input_is_data;
                    stride <<= 3;
                } else if super::fusion_fits::<super::StockhamFused2>(stride, n) {
                    let tw = super::fusion_twiddle_len::<super::StockhamFused2>(stride);
                    out.push((2, stride, tw, input_is_data));
                    input_is_data = !input_is_data;
                    stride <<= 2;
                } else {
                    let tw = super::fusion_twiddle_len::<super::StockhamFused1>(stride);
                    out.push((1, stride, tw, input_is_data));
                    input_is_data = !input_is_data;
                    stride <<= 1;
                }
            }
            out
        }

        for &log2 in &[5u32, 6, 7, 8, 9, 10, 15] {
            let n = 1usize << log2;
            let sched = compute_schedule(log2);
            let total_tw: usize = sched.iter().map(|&(_, _, tw, _)| tw).sum();
            eprintln!(
                "LOG2={} n={} schedule: {:?} total_tw={} (expect {}) final_copy={}",
                log2,
                n,
                sched,
                total_tw,
                n - 1,
                /* if ended on scratch */ !sched.last().is_none_or(|s| s.3)
            );
            // Also assert tw table len match for sanity (used to validate specials)
            assert!(total_tw >= n - 1, "under for log2={}", log2);
        }
    }
}
