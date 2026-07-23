//! Unrolled radix-1 triple stage specialised for n = 32.
//!
//! Extracted from `triple` so each size-specialised kernel occupies its own
//! leaf module; the shared element step stays in the parent.

use super::super::super::backend::StockhamAvxBackend;
use super::radix1_triple_do_one;

/// Unrolled (no loop) radix-1 triple for N=32 (eighth_n=4).
/// Explicit do_one calls for the 1-2 vector steps (step = COMPLEX_PER_VECTOR).
/// Called from per-LOG2 len32 body (first pass stride=1 triple) via stage_triple
/// when n==32 radix==1. Enables ILP / register tuning / DCE per monomorph
/// (Inner-Fn + structural specialization for md-worst PoT 32 from benchmark_results).
/// Zero extra cost for other sizes (not routed). Preserves exact same ops as general.
#[inline]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_radix1_n32_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, 32);
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

    // Explicit iters (no while, no k mut in hot path from len32). DCE on step at mono.
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
    if step <= 2 {
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
    // (avx512 per=4 on n32: 1 iter covered by first; falls to general in caller if needed)
}
