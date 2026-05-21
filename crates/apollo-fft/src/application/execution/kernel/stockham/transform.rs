//! Generic Stockham transform driver and fixed-size specializations.

use super::precision::{
    fusion_fits, fusion_twiddle_len, stockham_twiddle_subslice, stockham_twiddle_table_len,
    StockhamFusion, StockhamPrecision, StockhamTwiddleCursor,
};
use super::precision::{StockhamFused1, StockhamFused2, StockhamFused3, StockhamFused4};
// StockhamFused types imported below

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
