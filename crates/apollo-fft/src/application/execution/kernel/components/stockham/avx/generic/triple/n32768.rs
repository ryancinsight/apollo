//! Unrolled radix-1 triple stage specialised for n = 32768.
//!
//! Extracted from `triple` so each size-specialised kernel occupies its own
//! leaf module; the shared element step stays in the parent.

use super::super::super::backend::StockhamAvxBackend;
use super::radix1_triple_do_one;

/// 4x unrolled (explicit first-4 + 4x while for rest; step-uniform) radix-1 triple for N=32768 (eighth_n=4096).
/// First pass of len32768 (md f64 32768 2.75x worst remaining PoT after n1024 unroll).
/// Upgraded 2x->4x unroll of k loop (4 do_one per iter) to further reduce loop overhead and expose more ILP vs prior 2x/1.
/// Guards first-4 via sequential kk < eighth_n (covers step=2/4/8/16+ for avx/avx512 f32/f64; no under-execution).
/// Additive to prior n1024/n512/n256 + ZST mono/Cow. Preserves exact same do_one calls/ops/seq as general radix1 (no behavior change).
/// Zero extra cost for other sizes (not routed). No new allocs (reuses caller TL scratch + tw views).
#[inline(never)]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_radix1_n32768_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, 32768);
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

    // Explicit first up to 4 (0, s, 2s, 3s) using kk < bound (step-uniform for avx512 step=8/16 too).
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
    let mut kk = step;
    if kk < eighth_n {
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            kk,
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
        kk += step;
    }
    if kk < eighth_n {
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            kk,
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
        kk += step;
    }
    if kk < eighth_n {
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            kk,
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
        kk += step;
    }
    // 4x unrolled k loop for the rest (higher ILP / 1/4 loop count vs 1x for large n first pass of 32768).
    // For f64 step=2: 2048 iters total (first-4 + 2044 in 511 groups of 4).
    // For f32 step=4: 1024 iters total. Covers step=8+ (avx512) without missing iters.
    let mut k = kk;
    while k < eighth_n {
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            k,
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
        k += step;
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            k,
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
        k += step;
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            k,
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
        k += step;
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            k,
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
        k += step;
    }
}
