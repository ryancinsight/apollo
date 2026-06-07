use super::super::backend::StockhamAvxBackend;

#[inline(never)]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    radix: usize,
    groups: usize,
    first_twiddles: &[B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let quarter_groups = groups >> 2;
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(groups >= 8);
    debug_assert_eq!(groups & (groups - 1), 0);
    debug_assert_eq!(quarter_groups & (B::COMPLEX_PER_VECTOR - 1), 0);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= 2 * radix);
    debug_assert!(third_twiddles.len() >= 4 * radix);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();

    for j in 0..radix {
        let (w1re, w1im) = B::unpack_complex(*first_ptr.add(j));
        let (w2are, w2aim) = B::unpack_complex(*second_ptr.add(j));
        let (w2bre, w2bim) = B::unpack_complex(*second_ptr.add(j + radix));
        let (w3are, w3aim) = B::unpack_complex(*third_ptr.add(j));
        let (w3bre, w3bim) = B::unpack_complex(*third_ptr.add(j + radix));
        let (w3cre, w3cim) = B::unpack_complex(*third_ptr.add(j + 2 * radix));
        let (w3dre, w3dim) = B::unpack_complex(*third_ptr.add(j + 3 * radix));

        let w1r = B::set1_real(w1re);
        let w1i = B::set1_real(w1im);
        let w2ar = B::set1_real(w2are);
        let w2ai = B::set1_real(w2aim);
        let w2br = B::set1_real(w2bre);
        let w2bi = B::set1_real(w2bim);
        let w3ar = B::set1_real(w3are);
        let w3ai = B::set1_real(w3aim);
        let w3br = B::set1_real(w3bre);
        let w3bi = B::set1_real(w3bim);
        let w3cr = B::set1_real(w3cre);
        let w3ci = B::set1_real(w3cim);
        let w3dr = B::set1_real(w3dre);
        let w3di = B::set1_real(w3dim);

        let src_base = j * groups * 2;
        let dst_base = j * quarter_groups;
        let mut k = 0usize;
        while k < quarter_groups {
            let x0 = B::loadu_complex(src_ptr.add(src_base + k));
            let x2 = B::loadu_complex(src_ptr.add(src_base + 2 * quarter_groups + k));
            let x4 = B::cmul(
                w1r,
                w1i,
                B::loadu_complex(src_ptr.add(src_base + groups + k)),
            );
            let x6 = B::cmul(
                w1r,
                w1i,
                B::loadu_complex(src_ptr.add(src_base + groups + 2 * quarter_groups + k)),
            );

            let s0 = B::add(x0, x4);
            let s2 = B::add(x2, x6);
            let d0 = B::sub(x0, x4);
            let d2 = B::sub(x2, x6);

            let t2 = B::cmul(w2ar, w2ai, s2);
            let p0 = B::add(s0, t2);
            let p4 = B::sub(s0, t2);

            let x1 = B::loadu_complex(src_ptr.add(src_base + quarter_groups + k));
            let x3 = B::loadu_complex(src_ptr.add(src_base + 3 * quarter_groups + k));
            let x5 = B::cmul(
                w1r,
                w1i,
                B::loadu_complex(src_ptr.add(src_base + groups + quarter_groups + k)),
            );
            let x7 = B::cmul(
                w1r,
                w1i,
                B::loadu_complex(src_ptr.add(src_base + groups + 3 * quarter_groups + k)),
            );

            let s1 = B::add(x1, x5);
            let s3 = B::add(x3, x7);
            let d1 = B::sub(x1, x5);
            let d3 = B::sub(x3, x7);

            let t3 = B::cmul(w2ar, w2ai, s3);
            let p1 = B::add(s1, t3);
            let p5 = B::sub(s1, t3);

            let out_base = dst_base + k;
            let q0 = B::cmul(w3ar, w3ai, p1);
            let q2 = B::cmul(w3cr, w3ci, p5);
            B::storeu_complex(dst_ptr.add(out_base), B::add(p0, q0));
            B::storeu_complex(dst_ptr.add(half_n + out_base), B::sub(p0, q0));
            B::storeu_complex(dst_ptr.add(quarter_n + out_base), B::add(p4, q2));
            B::storeu_complex(dst_ptr.add(half_n + quarter_n + out_base), B::sub(p4, q2));

            let u2 = B::cmul(w2br, w2bi, d2);
            let u3 = B::cmul(w2br, w2bi, d3);
            let p2 = B::add(d0, u2);
            let p3 = B::add(d1, u3);
            let p6 = B::sub(d0, u2);
            let p7 = B::sub(d1, u3);
            let q1 = B::cmul(w3br, w3bi, p3);
            let q3 = B::cmul(w3dr, w3di, p7);
            B::storeu_complex(dst_ptr.add(eighth_n + out_base), B::add(p2, q1));
            B::storeu_complex(dst_ptr.add(half_n + eighth_n + out_base), B::sub(p2, q1));
            B::storeu_complex(dst_ptr.add(quarter_n + eighth_n + out_base), B::add(p6, q3));
            B::storeu_complex(
                dst_ptr.add(half_n + quarter_n + eighth_n + out_base),
                B::sub(p6, q3),
            );
            k += B::COMPLEX_PER_VECTOR;
        }
    }
}

#[inline]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_low_live_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    radix: usize,
    groups: usize,
    first_twiddles: &[B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let quarter_groups = groups >> 2;
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(groups >= 8);
    debug_assert_eq!(groups & (groups - 1), 0);
    debug_assert_eq!(quarter_groups & (B::COMPLEX_PER_VECTOR - 1), 0);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= 2 * radix);
    debug_assert!(third_twiddles.len() >= 4 * radix);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();

    for j in 0..radix {
        let (w1re, w1im) = B::unpack_complex(*first_ptr.add(j));
        let (w2are, w2aim) = B::unpack_complex(*second_ptr.add(j));
        let (w2bre, w2bim) = B::unpack_complex(*second_ptr.add(j + radix));
        let (w3are, w3aim) = B::unpack_complex(*third_ptr.add(j));
        let (w3bre, w3bim) = B::unpack_complex(*third_ptr.add(j + radix));
        let (w3cre, w3cim) = B::unpack_complex(*third_ptr.add(j + 2 * radix));
        let (w3dre, w3dim) = B::unpack_complex(*third_ptr.add(j + 3 * radix));

        let w1r = B::set1_real(w1re);
        let w1i = B::set1_real(w1im);
        let w2ar = B::set1_real(w2are);
        let w2ai = B::set1_real(w2aim);
        let w2br = B::set1_real(w2bre);
        let w2bi = B::set1_real(w2bim);
        let w3ar = B::set1_real(w3are);
        let w3ai = B::set1_real(w3aim);
        let w3br = B::set1_real(w3bre);
        let w3bi = B::set1_real(w3bim);
        let w3cr = B::set1_real(w3cre);
        let w3ci = B::set1_real(w3cim);
        let w3dr = B::set1_real(w3dre);
        let w3di = B::set1_real(w3dim);
        let src_base = j * groups * 2;
        let dst_base = j * quarter_groups;
        let mut k = 0usize;
        while k < quarter_groups {
            let x0 = B::loadu_complex(src_ptr.add(src_base + k));
            let x2 = B::loadu_complex(src_ptr.add(src_base + 2 * quarter_groups + k));
            let x4 = B::cmul(
                w1r,
                w1i,
                B::loadu_complex(src_ptr.add(src_base + groups + k)),
            );
            let x6 = B::cmul(
                w1r,
                w1i,
                B::loadu_complex(src_ptr.add(src_base + groups + 2 * quarter_groups + k)),
            );

            let s0 = B::add(x0, x4);
            let s2 = B::add(x2, x6);
            let d0 = B::sub(x0, x4);
            let d2 = B::sub(x2, x6);

            let t2 = B::cmul(w2ar, w2ai, s2);
            let p0 = B::add(s0, t2);
            let p4 = B::sub(s0, t2);

            let x1 = B::loadu_complex(src_ptr.add(src_base + quarter_groups + k));
            let x3 = B::loadu_complex(src_ptr.add(src_base + 3 * quarter_groups + k));
            let x5 = B::cmul(
                w1r,
                w1i,
                B::loadu_complex(src_ptr.add(src_base + groups + quarter_groups + k)),
            );
            let x7 = B::cmul(
                w1r,
                w1i,
                B::loadu_complex(src_ptr.add(src_base + groups + 3 * quarter_groups + k)),
            );
            let s1 = B::add(x1, x5);
            let s3 = B::add(x3, x7);
            let d1 = B::sub(x1, x5);
            let d3 = B::sub(x3, x7);

            let t3 = B::cmul(w2ar, w2ai, s3);
            let p1 = B::add(s1, t3);
            let p5 = B::sub(s1, t3);

            let out_base = dst_base + k;
            let q0 = B::cmul(w3ar, w3ai, p1);
            let q2 = B::cmul(w3cr, w3ci, p5);
            B::storeu_complex(dst_ptr.add(out_base), B::add(p0, q0));
            B::storeu_complex(dst_ptr.add(half_n + out_base), B::sub(p0, q0));
            B::storeu_complex(dst_ptr.add(quarter_n + out_base), B::add(p4, q2));
            B::storeu_complex(dst_ptr.add(half_n + quarter_n + out_base), B::sub(p4, q2));

            let u2 = B::cmul(w2br, w2bi, d2);
            let u3 = B::cmul(w2br, w2bi, d3);
            let p2 = B::add(d0, u2);
            let p3 = B::add(d1, u3);
            let p6 = B::sub(d0, u2);
            let p7 = B::sub(d1, u3);
            let q1 = B::cmul(w3br, w3bi, p3);
            let q3 = B::cmul(w3dr, w3di, p7);
            B::storeu_complex(dst_ptr.add(eighth_n + out_base), B::add(p2, q1));
            B::storeu_complex(dst_ptr.add(half_n + eighth_n + out_base), B::sub(p2, q1));
            B::storeu_complex(dst_ptr.add(quarter_n + eighth_n + out_base), B::add(p6, q3));
            B::storeu_complex(
                dst_ptr.add(half_n + quarter_n + eighth_n + out_base),
                B::sub(p6, q3),
            );
            k += B::COMPLEX_PER_VECTOR;
        }
    }
}

#[inline]
unsafe fn radix1_triple_do_one<B: StockhamAvxBackend>(
    src_ptr: *const B::Complex,
    dst_ptr: *mut B::Complex,
    k: usize,
    eighth_n: usize,
    half_n: usize,
    quarter_n: usize,
    w2_quarter_turn_sign: B::Real,
    w3br: B::Vector,
    w3bi: B::Vector,
    w3dr: B::Vector,
    w3di: B::Vector,
    w3_quarter_turn_sign: B::Real,
) {
    let x0 = B::loadu_complex(src_ptr.add(k));
    let x2 = B::loadu_complex(src_ptr.add(2 * eighth_n + k));
    let x4 = B::loadu_complex(src_ptr.add(4 * eighth_n + k));
    let x6 = B::loadu_complex(src_ptr.add(6 * eighth_n + k));

    let s0 = B::add(x0, x4);
    let s2 = B::add(x2, x6);
    let d0 = B::sub(x0, x4);
    let d2 = B::sub(x2, x6);

    let u2 = B::rotate_quarter_turn(d2, w2_quarter_turn_sign);
    let p0 = B::add(s0, s2);
    let p2 = B::add(d0, u2);
    let p4 = B::sub(s0, s2);
    let p6 = B::sub(d0, u2);

    let x1 = B::loadu_complex(src_ptr.add(eighth_n + k));
    let x3 = B::loadu_complex(src_ptr.add(3 * eighth_n + k));
    let x5 = B::loadu_complex(src_ptr.add(5 * eighth_n + k));
    let x7 = B::loadu_complex(src_ptr.add(7 * eighth_n + k));

    let s1 = B::add(x1, x5);
    let s3 = B::add(x3, x7);
    let d1 = B::sub(x1, x5);
    let d3 = B::sub(x3, x7);
    let u3 = B::rotate_quarter_turn(d3, w2_quarter_turn_sign);
    let p1 = B::add(s1, s3);
    let p3 = B::add(d1, u3);
    let p5 = B::sub(s1, s3);
    let p7 = B::sub(d1, u3);

    let q2 = B::rotate_quarter_turn(p5, w3_quarter_turn_sign);
    B::storeu_complex(dst_ptr.add(k), B::add(p0, p1));
    B::storeu_complex(dst_ptr.add(half_n + k), B::sub(p0, p1));
    B::storeu_complex(dst_ptr.add(quarter_n + k), B::add(p4, q2));
    B::storeu_complex(dst_ptr.add(half_n + quarter_n + k), B::sub(p4, q2));

    let q1 = B::cmul(w3br, w3bi, p3);
    let q3 = B::cmul(w3dr, w3di, p7);
    B::storeu_complex(dst_ptr.add(eighth_n + k), B::add(p2, q1));
    B::storeu_complex(dst_ptr.add(half_n + eighth_n + k), B::sub(p2, q1));
    B::storeu_complex(dst_ptr.add(quarter_n + eighth_n + k), B::add(p6, q3));
    B::storeu_complex(
        dst_ptr.add(half_n + quarter_n + eighth_n + k),
        B::sub(p6, q3),
    );
}

#[inline(never)]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_radix1_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(n >= B::COMPLEX_PER_VECTOR * 8);
    debug_assert_eq!(n & (n - 1), 0);
    debug_assert!(second_twiddles.len() >= 2);
    debug_assert!(third_twiddles.len() >= 4);
    debug_assert_eq!(eighth_n & (B::COMPLEX_PER_VECTOR - 1), 0);

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
    let mut k = 0usize;
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
        k += B::COMPLEX_PER_VECTOR;
    }
}

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

/// Unrolled (no loop) radix-1 triple for N=64 (eighth_n=8).
/// Explicit do_one calls for the vector steps (step = COMPLEX_PER_VECTOR: 2 for f64, 4 for f32).
/// Called from stage_triple when n==64 radix==1 (first pass of len64, md-worst PoT 64).
/// Enables ILP / register tuning / DCE per monomorph (Inner-Fn + structural const LOG2).
/// Additive zero-cost to n32 special + ZST wiring; preserves exact same ops as general radix1 path.
/// Zero extra cost for other sizes (not routed). Targets 64 ratios from benchmark_results.md.
#[inline]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_radix1_n64_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, 64);
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

    // Explicit iters (no while, no k mut in hot path from len64). DCE on step at mono.
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
    if step <= 4 {
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
    }
}

/// Unrolled (no loop) radix-1 triple for N=128 (eighth_n=16).
/// Explicit do_one calls for the vector steps (step = COMPLEX_PER_VECTOR: 2 for f64, 4 for f32).
/// Called from stage_triple when n==128 radix==1 (first pass of len128).
/// Enables ILP / DCE per monomorph for remaining PoT 128 (md ratios ~1.27x f32).
/// Additive to n32/n64 + ZST LOG2=7. Zero extra cost for other sizes. Preserves exact ops.
#[inline(never)]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_radix1_n128_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, 128);
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

    // Explicit iters (no while, no k mut). DCE on step at mono.
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
    if step <= 8 {
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
    } else if step == 4 {
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
    }
}

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

/// Unrolled (no loop) radix-1 triple for N=512 (eighth_n=64).
/// Explicit do_one calls for the vector steps (step = COMPLEX_PER_VECTOR: 2 for f64 ~32 iters, 4 for f32 ~16 iters).
/// Called from stage_triple when n==512 radix==1 (first pass of len512, and p=512 pads in f32 rader bluestein for n113/257 etc).
/// Enables ILP/DCE per mono for PoT 512/1024/32768 structure (md f64 512 1.241x, 32768 2.75x controlling; 32/64/128/256 benefit indirectly via structure).
/// Additive to n256/128/64/32 + ZST. Zero extra for other sizes. Preserves exact ops as general radix1.
#[inline(never)]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_radix1_n512_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, 512);
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

    // Explicit iters no while/mut k. DCE on step at mono for n=512 mono.
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
    if step <= 32 {
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
        // 32 iters total for f64 per=2
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
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            16 * step,
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
            17 * step,
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
            18 * step,
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
            19 * step,
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
            20 * step,
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
            21 * step,
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
            22 * step,
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
            23 * step,
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
            24 * step,
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
            25 * step,
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
            26 * step,
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
            27 * step,
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
            28 * step,
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
            29 * step,
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
            30 * step,
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
            31 * step,
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
        // 16 iters for f32 per=4
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
    }
}

/// Unrolled (no loop) radix-1 triple for N=1024 (eighth_n=128).
/// Explicit do_one calls for the vector steps (step = COMPLEX_PER_VECTOR: 2 for f64 ~64 iters, 4 for f32 ~32 iters).
/// Called from stage_triple when n==1024 radix==1 (first pass of len1024, and p=1024 pads in f32 rader bluestein).
/// Enables ILP/DCE per mono for PoT 1024/32768 structure (md 32768 f64 2.75x controlling; 4x unroll first-pass extends 512/256 structure).
/// Additive to n512+ + ZST. Zero extra for other. Preserves exact ops as general radix1. Advances per-LOG2 + f32 sub for pads.
#[inline(never)]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple_radix1_n1024_avx_fma<B: StockhamAvxBackend>(
    src: &[B::Complex],
    dst: &mut [B::Complex],
    second_twiddles: &[B::Complex],
    third_twiddles: &[B::Complex],
) {
    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, 1024);
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

    // Explicit iters no while/mut k. DCE on step at mono for n=1024 mono.
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
    if step <= 64 {
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
        // 64 iters total for f64 per=2
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
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            16 * step,
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
            17 * step,
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
            18 * step,
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
            19 * step,
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
            20 * step,
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
            21 * step,
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
            22 * step,
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
            23 * step,
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
            24 * step,
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
            25 * step,
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
            26 * step,
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
            27 * step,
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
            28 * step,
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
            29 * step,
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
            30 * step,
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
            31 * step,
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
            32 * step,
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
            33 * step,
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
            34 * step,
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
            35 * step,
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
            36 * step,
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
            37 * step,
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
            38 * step,
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
            39 * step,
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
            40 * step,
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
            41 * step,
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
            42 * step,
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
            43 * step,
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
            44 * step,
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
            45 * step,
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
            46 * step,
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
            47 * step,
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
            48 * step,
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
            49 * step,
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
            50 * step,
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
            51 * step,
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
            52 * step,
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
            53 * step,
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
            54 * step,
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
            55 * step,
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
            56 * step,
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
            57 * step,
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
            58 * step,
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
            59 * step,
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
            60 * step,
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
            61 * step,
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
            62 * step,
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
            63 * step,
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
        // 32 iters for f32 per=4
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
        radix1_triple_do_one::<B>(
            src_ptr,
            dst_ptr,
            16 * step,
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
            17 * step,
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
            18 * step,
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
            19 * step,
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
            20 * step,
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
            21 * step,
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
            22 * step,
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
            23 * step,
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
            24 * step,
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
            25 * step,
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
            26 * step,
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
            27 * step,
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
            28 * step,
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
            29 * step,
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
            30 * step,
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
            31 * step,
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
