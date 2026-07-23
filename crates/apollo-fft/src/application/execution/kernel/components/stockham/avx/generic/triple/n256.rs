//! Unrolled radix-1 triple stage specialised for n = 256.
//!
//! Extracted from `triple` so each size-specialised kernel occupies its own
//! leaf module; the shared element step stays in the parent.

use super::super::super::backend::StockhamAvxBackend;
use super::radix1_triple_do_one;

/// Unrolled (no loop) radix-1 triple for N=256 (eighth_n=32).
/// Explicit do_one calls for the vector steps (step = COMPLEX_PER_VECTOR: 2 for f64 ~16 iters, 4 for f32 ~8 iters).
/// Called from stage_triple when n==256 radix==1 (first pass of len256, and p=256 pads in f32 rader bluestein for n113 etc).
/// Enables ILP/DCE per mono for PoT 256/512/32768 structure (md f32 256 1.137x, 32768 2.75x controlling after 4x unroll; 32/64 also benefit indirectly).
/// Additive to 128/64/32 + ZST. Zero extra for other. Preserves exact ops as general radix1.
#[inline(never)]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_radix1_n256_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, 256);
    debug_assert!(second_twiddles.len() >= 2);
    debug_assert!(third_twiddles.len() >= 4);

    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();
    let (_w2bre, w2bim) = B::unpack_complex(*second_ptr.add(1));
    let (w3bre, w3bim) = B::unpack_complex(*third_ptr.add(1));
    let (_w3cre, w3cim) = B::unpack_complex(*third_ptr.add(2));
    let (w3dre, w3dim) = B::unpack_complex(*third_ptr.add(3));
    let w3br = B::set1_real(w3bre);
    let w3bi = B::set1_real(w3bim);
    let w3dr = B::set1_real(w3dre);
    let w3di = B::set1_real(w3dim);
    let w2_quarter_turn_sign = w2bim;
    let w3_quarter_turn_sign = w3cim;

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    // Explicit iters no while/mut k. DCE on step at mono for n=256 mono.
    radix1_triple_do_one::<B>(
        src_ptr,
        dst_ptr,
        0,
        eighth_n,
        half_n,
        quarter_n,
        w2_quarter_turn_sign,
        w3br,
        w3bi,
        w3dr,
        w3di,
        w3_quarter_turn_sign,
    );
    let step = B::COMPLEX_PER_VECTOR;
    if step <= 16 {
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
    }
    if step == 2 {
        // 16 iters total for f64 per=2
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            2 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            3 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            4 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            5 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            6 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            7 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            8 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            9 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            10 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            11 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            12 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            13 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            14 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            15 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
    } else if step == 4 {
        // 8 iters for f32 per=4
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            2 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            3 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            4 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            5 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            6 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            7 * step,
            eighth_n,
            half_n,
            quarter_n,
            w2_quarter_turn_sign,
            w3br,
            w3bi,
            w3dr,
            w3di,
            w3_quarter_turn_sign,
        );
    }
}
