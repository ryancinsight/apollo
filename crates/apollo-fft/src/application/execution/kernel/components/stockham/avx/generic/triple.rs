use super::super::backend::StockhamAvxBackend;

#[inline]
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
        k += B::COMPLEX_PER_VECTOR;
    }
}
