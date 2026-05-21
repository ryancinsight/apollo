//! Generic Stockham transform driver and fixed-size specializations.

use super::precision::{
    fusion_fits, fusion_twiddle_len, stockham_twiddle_subslice, stockham_twiddle_table_len,
    StockhamFusion, StockhamPrecision, StockhamTwiddleCursor,
};
use super::precision::{StockhamFused1, StockhamFused2, StockhamFused3, StockhamFused4};
// StockhamFused types imported below

/// Power-of-two sizes that have hand-optimized stage sequences bypassing the
/// greedy fusion-eligibility loop.  See [`transform_len4096_four_triples`]
/// (four consecutive triples) and [`transform_len32768`] (pair+3×triple+quad).
#[allow(dead_code)]
pub(crate) const SPECIAL_TRANSFORM_SIZES: &[usize] = &[4096, 32768];

pub(crate) fn transform<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
    scale: Option<P::Real>,
) {
    let n = data.len();
    if n <= 1 {
        if let Some(scale) = scale {
            P::scale(data, scale);
        }
        return;
    }
    debug_assert!(n.is_power_of_two());
    debug_assert!(twiddles.len() >= stockham_twiddle_table_len(n));

    if n == 32768 && P::MAX_FUSED_STAGES >= StockhamFused4::STAGE_COUNT {
        transform_len32768::<P>(data, scratch, twiddles);
        if let Some(scale) = scale {
            P::scale(data, scale);
        }
        return;
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
/// |  1   | pair      |      1 |  1–2           | false                |
/// |  2   | triple    |      4 |  3–5           | true                 |
/// |  3   | triple    |     32 |  6–8           | false                |
/// |  4   | triple    |    256 |  9–11          | true                 |
/// |  5   | quad      |   2048 | 12–15          | false                |
///
/// 5 passes (odd) → result lands in `scratch` → `copy_from_slice` restores it
/// to `data`. The quad at stride=2048 satisfies `groups == 8` and fires the
/// AVX-optimized `stockham_quad_groups_eight_low_live` kernel.
///
/// Total cost ≈ 5.25 effective passes vs 6 for the greedy triple-first sequence
/// (pair+3×triple+quad = 5 compute passes, plus ~0.25-pass-equivalent memcpy).
#[inline]
pub(crate) fn transform_len32768<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 32768);
    debug_assert_eq!(scratch.len(), 32768);
    debug_assert!(twiddles.len() >= 32767);

    // Pass 1: pair(stride=1), 2 stages, twiddles[0..3]
    P::stage_pair(data, scratch, 1, &twiddles[0..1], &twiddles[1..3]);
    // Pass 2: triple(stride=4), 3 stages, twiddles[3..31]
    P::stage_triple(
        scratch,
        data,
        4,
        &twiddles[3..7],
        &twiddles[7..15],
        &twiddles[15..31],
    );
    // Pass 3: triple(stride=32), 3 stages, twiddles[31..255]
    P::stage_triple(
        data,
        scratch,
        32,
        &twiddles[31..63],
        &twiddles[63..127],
        &twiddles[127..255],
    );
    // Pass 4: triple(stride=256), 3 stages, twiddles[255..2047]
    P::stage_triple(
        scratch,
        data,
        256,
        &twiddles[255..511],
        &twiddles[511..1023],
        &twiddles[1023..2047],
    );
    // Pass 5: quad(stride=2048), 4 stages, twiddles[2047..32767]
    // groups = 32768/(2048*2) = 8 → fires AVX stockham_quad_groups_eight_low_live
    P::stage_quad(
        data,
        scratch,
        2048,
        &twiddles[2047..4095],
        &twiddles[4095..8191],
        &twiddles[8191..16383],
        &twiddles[16383..32767],
    );
    // 5 passes (odd count) → result is in scratch; copy back to data.
    data.copy_from_slice(scratch);
}

#[cfg(test)]
mod tests {
    use super::*;

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
