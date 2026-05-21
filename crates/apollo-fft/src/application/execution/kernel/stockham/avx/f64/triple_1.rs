use super::fixed::{avx_rotate_quarter_turn, cmul_vec64};
use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple64_radix1_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    second_twiddles: &[Complex64],
    third_twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_set_pd, _mm256_storeu_pd,
        _mm256_sub_pd,
    };

    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(n >= 16);
    debug_assert_eq!(n & (n - 1), 0);
    debug_assert!(second_twiddles.len() >= 2);
    debug_assert!(third_twiddles.len() >= 4);
    debug_assert_eq!(eighth_n & 1, 0);

    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();
    let w2b = *second_ptr.add(1);
    let w3b = *third_ptr.add(1);
    let w3c = *third_ptr.add(2);
    let w3d = *third_ptr.add(3);
    let w3br = _mm256_set1_pd(w3b.re);
    let w3bi = _mm256_set1_pd(w3b.im);
    let w3dr = _mm256_set1_pd(w3d.re);
    let w3di = _mm256_set1_pd(w3d.im);
    let w2_quarter_turn_mask = if w2b.im > 0.0 {
        _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
    } else {
        _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
    };
    let w3_quarter_turn_mask = if w3c.im > 0.0 {
        _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
    } else {
        _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
    };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mut k = 0usize;
    while k < eighth_n {
        let x0 = _mm256_loadu_pd(src_ptr.add(k).cast::<f64>());
        let x2 = _mm256_loadu_pd(src_ptr.add(2 * eighth_n + k).cast::<f64>());
        let x4 = _mm256_loadu_pd(src_ptr.add(4 * eighth_n + k).cast::<f64>());
        let x6 = _mm256_loadu_pd(src_ptr.add(6 * eighth_n + k).cast::<f64>());

        let s0 = _mm256_add_pd(x0, x4);
        let s2 = _mm256_add_pd(x2, x6);
        let d0 = _mm256_sub_pd(x0, x4);
        let d2 = _mm256_sub_pd(x2, x6);

        let u2 = avx_rotate_quarter_turn(d2, w2_quarter_turn_mask);
        let p0 = _mm256_add_pd(s0, s2);
        let p2 = _mm256_add_pd(d0, u2);
        let p4 = _mm256_sub_pd(s0, s2);
        let p6 = _mm256_sub_pd(d0, u2);

        let x1 = _mm256_loadu_pd(src_ptr.add(eighth_n + k).cast::<f64>());
        let x3 = _mm256_loadu_pd(src_ptr.add(3 * eighth_n + k).cast::<f64>());
        let x5 = _mm256_loadu_pd(src_ptr.add(5 * eighth_n + k).cast::<f64>());
        let x7 = _mm256_loadu_pd(src_ptr.add(7 * eighth_n + k).cast::<f64>());

        let s1 = _mm256_add_pd(x1, x5);
        let s3 = _mm256_add_pd(x3, x7);
        let d1 = _mm256_sub_pd(x1, x5);
        let d3 = _mm256_sub_pd(x3, x7);
        let u3 = avx_rotate_quarter_turn(d3, w2_quarter_turn_mask);
        let p1 = _mm256_add_pd(s1, s3);
        let p3 = _mm256_add_pd(d1, u3);
        let p5 = _mm256_sub_pd(s1, s3);
        let p7 = _mm256_sub_pd(d1, u3);

        let q2 = avx_rotate_quarter_turn(p5, w3_quarter_turn_mask);
        _mm256_storeu_pd(dst_ptr.add(k).cast::<f64>(), _mm256_add_pd(p0, p1));
        _mm256_storeu_pd(dst_ptr.add(half_n + k).cast::<f64>(), _mm256_sub_pd(p0, p1));
        _mm256_storeu_pd(
            dst_ptr.add(quarter_n + k).cast::<f64>(),
            _mm256_add_pd(p4, q2),
        );
        _mm256_storeu_pd(
            dst_ptr.add(half_n + quarter_n + k).cast::<f64>(),
            _mm256_sub_pd(p4, q2),
        );

        let q1 = cmul_vec64(w3br, w3bi, p3);
        let q3 = cmul_vec64(w3dr, w3di, p7);
        _mm256_storeu_pd(
            dst_ptr.add(eighth_n + k).cast::<f64>(),
            _mm256_add_pd(p2, q1),
        );
        _mm256_storeu_pd(
            dst_ptr.add(half_n + eighth_n + k).cast::<f64>(),
            _mm256_sub_pd(p2, q1),
        );
        _mm256_storeu_pd(
            dst_ptr.add(quarter_n + eighth_n + k).cast::<f64>(),
            _mm256_add_pd(p6, q3),
        );
        _mm256_storeu_pd(
            dst_ptr.add(half_n + quarter_n + eighth_n + k).cast::<f64>(),
            _mm256_sub_pd(p6, q3),
        );
        k += 2;
    }
}

/// Fuses the first three f32 Stockham stages when the incoming radix is one.
///
/// For `radix = 1`, the first-stage twiddle and the lower second/third-stage
/// twiddles are exactly one. The upper second-stage twiddle is a quarter-turn,
/// and the third-stage `W8^2` branch is also a quarter-turn. Direct substitution
/// removes those complex products and keeps only the `W8^1` and `W8^3` FMA
/// products. The function is the f32 native-precision analogue of
/// `stage_triple64_radix1_avx_fma`; it vectorizes across independent `k`
/// instances and writes the same eight Stockham autosort bands.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple64_quarter_groups_one_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    radix: usize,
    first_twiddles: &[Complex64],
    second_twiddles: &[Complex64],
    third_twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_extractf128_pd, _mm256_loadu_pd, _mm256_permute2f128_pd,
        _mm256_set1_pd, _mm256_set_pd, _mm256_sub_pd, _mm_storeu_pd,
    };

    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, radix << 3);
    debug_assert_eq!(dst.len(), n);
    debug_assert!(radix >= 1);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= 2 * radix);
    debug_assert!(third_twiddles.len() >= 4 * radix);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();

    for j in 0..radix {
        let w1 = *first_ptr.add(j);
        let w2a = *second_ptr.add(j);
        let w2b = *second_ptr.add(j + radix);
        let w3a = *third_ptr.add(j);
        let w3b = *third_ptr.add(j + radix);
        let w3c = *third_ptr.add(j + 2 * radix);
        let w3d = *third_ptr.add(j + 3 * radix);

        let w1r = _mm256_set1_pd(w1.re);
        let w1i = _mm256_set1_pd(w1.im);

        let src_base = j * 8;

        let v01 = _mm256_loadu_pd(src_ptr.add(src_base).cast::<f64>());
        let v23 = _mm256_loadu_pd(src_ptr.add(src_base + 2).cast::<f64>());
        let v45 = _mm256_loadu_pd(src_ptr.add(src_base + 4).cast::<f64>());
        let v67 = _mm256_loadu_pd(src_ptr.add(src_base + 6).cast::<f64>());

        let tw45 = cmul_vec64(w1r, w1i, v45);
        let tw67 = cmul_vec64(w1r, w1i, v67);

        let s01 = _mm256_add_pd(v01, tw45);
        let s23 = _mm256_add_pd(v23, tw67);
        let d01 = _mm256_sub_pd(v01, tw45);
        let d23 = _mm256_sub_pd(v23, tw67);

        let w2ar = _mm256_set1_pd(w2a.re);
        let w2ai = _mm256_set1_pd(w2a.im);
        let t23 = cmul_vec64(w2ar, w2ai, s23);

        let w2br = _mm256_set1_pd(w2b.re);
        let w2bi = _mm256_set1_pd(w2b.im);
        let u23 = cmul_vec64(w2br, w2bi, d23);

        let p01 = _mm256_add_pd(s01, t23);
        let p45 = _mm256_sub_pd(s01, t23);
        let p23 = _mm256_add_pd(d01, u23);
        let p67 = _mm256_sub_pd(d01, u23);

        let p13 = _mm256_permute2f128_pd(p01, p23, 0x31);
        let w3ab_r = _mm256_set_pd(w3b.re, w3b.re, w3a.re, w3a.re);
        let w3ab_i = _mm256_set_pd(w3b.im, w3b.im, w3a.im, w3a.im);
        let q01 = cmul_vec64(w3ab_r, w3ab_i, p13);

        let p57 = _mm256_permute2f128_pd(p45, p67, 0x31);
        let w3cd_r = _mm256_set_pd(w3d.re, w3d.re, w3c.re, w3c.re);
        let w3cd_i = _mm256_set_pd(w3d.im, w3d.im, w3c.im, w3c.im);
        let q23 = cmul_vec64(w3cd_r, w3cd_i, p57);

        let p02 = _mm256_permute2f128_pd(p01, p23, 0x20);
        let out02 = _mm256_add_pd(p02, q01);
        let out13 = _mm256_sub_pd(p02, q01);

        let p46 = _mm256_permute2f128_pd(p45, p67, 0x20);
        let out46 = _mm256_add_pd(p46, q23);
        let out57 = _mm256_sub_pd(p46, q23);

        let out_base = j;
        _mm_storeu_pd(
            dst_ptr.add(out_base).cast::<f64>(),
            _mm256_extractf128_pd(out02, 0),
        );
        _mm_storeu_pd(
            dst_ptr.add(half_n + out_base).cast::<f64>(),
            _mm256_extractf128_pd(out13, 0),
        );
        _mm_storeu_pd(
            dst_ptr.add(eighth_n + out_base).cast::<f64>(),
            _mm256_extractf128_pd(out02, 1),
        );
        _mm_storeu_pd(
            dst_ptr.add(half_n + eighth_n + out_base).cast::<f64>(),
            _mm256_extractf128_pd(out13, 1),
        );

        _mm_storeu_pd(
            dst_ptr.add(quarter_n + out_base).cast::<f64>(),
            _mm256_extractf128_pd(out46, 0),
        );
        _mm_storeu_pd(
            dst_ptr.add(half_n + quarter_n + out_base).cast::<f64>(),
            _mm256_extractf128_pd(out57, 0),
        );
        _mm_storeu_pd(
            dst_ptr.add(quarter_n + eighth_n + out_base).cast::<f64>(),
            _mm256_extractf128_pd(out46, 1),
        );
        _mm_storeu_pd(
            dst_ptr
                .add(half_n + quarter_n + eighth_n + out_base)
                .cast::<f64>(),
            _mm256_extractf128_pd(out57, 1),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple64_low_live_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    radix: usize,
    groups: usize,
    first_twiddles: &[Complex64],
    second_twiddles: &[Complex64],
    third_twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };

    let n = src.len();
    let quarter_groups = groups >> 2;
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(groups >= 8);
    debug_assert_eq!(groups & (groups - 1), 0);
    debug_assert_eq!(quarter_groups & 1, 0);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= 2 * radix);
    debug_assert!(third_twiddles.len() >= 4 * radix);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();

    for j in 0..radix {
        let w1 = *first_ptr.add(j);
        let w2a = *second_ptr.add(j);
        let w2b = *second_ptr.add(j + radix);
        let w3a = *third_ptr.add(j);
        let w3b = *third_ptr.add(j + radix);
        let w3c = *third_ptr.add(j + 2 * radix);
        let w3d = *third_ptr.add(j + 3 * radix);
        let w1r = _mm256_set1_pd(w1.re);
        let w1i = _mm256_set1_pd(w1.im);
        let w2ar = _mm256_set1_pd(w2a.re);
        let w2ai = _mm256_set1_pd(w2a.im);
        let w2br = _mm256_set1_pd(w2b.re);
        let w2bi = _mm256_set1_pd(w2b.im);
        let w3ar = _mm256_set1_pd(w3a.re);
        let w3ai = _mm256_set1_pd(w3a.im);
        let w3br = _mm256_set1_pd(w3b.re);
        let w3bi = _mm256_set1_pd(w3b.im);
        let w3cr = _mm256_set1_pd(w3c.re);
        let w3ci = _mm256_set1_pd(w3c.im);
        let w3dr = _mm256_set1_pd(w3d.re);
        let w3di = _mm256_set1_pd(w3d.im);
        let src_base = j * groups * 2;
        let dst_base = j * quarter_groups;
        let mut k = 0usize;
        while k < quarter_groups {
            let x0 = _mm256_loadu_pd(src_ptr.add(src_base + k).cast::<f64>());
            let x2 = _mm256_loadu_pd(src_ptr.add(src_base + 2 * quarter_groups + k).cast::<f64>());
            let x4 = cmul_vec64(
                w1r,
                w1i,
                _mm256_loadu_pd(src_ptr.add(src_base + groups + k).cast::<f64>()),
            );
            let x6 = cmul_vec64(
                w1r,
                w1i,
                _mm256_loadu_pd(
                    src_ptr
                        .add(src_base + groups + 2 * quarter_groups + k)
                        .cast::<f64>(),
                ),
            );

            let s0 = _mm256_add_pd(x0, x4);
            let s2 = _mm256_add_pd(x2, x6);
            let d0 = _mm256_sub_pd(x0, x4);
            let d2 = _mm256_sub_pd(x2, x6);

            let t2 = cmul_vec64(w2ar, w2ai, s2);
            let p0 = _mm256_add_pd(s0, t2);
            let p4 = _mm256_sub_pd(s0, t2);
            let x1 = _mm256_loadu_pd(src_ptr.add(src_base + quarter_groups + k).cast::<f64>());
            let x3 = _mm256_loadu_pd(src_ptr.add(src_base + 3 * quarter_groups + k).cast::<f64>());
            let x5 = cmul_vec64(
                w1r,
                w1i,
                _mm256_loadu_pd(
                    src_ptr
                        .add(src_base + groups + quarter_groups + k)
                        .cast::<f64>(),
                ),
            );
            let x7 = cmul_vec64(
                w1r,
                w1i,
                _mm256_loadu_pd(
                    src_ptr
                        .add(src_base + groups + 3 * quarter_groups + k)
                        .cast::<f64>(),
                ),
            );
            let s1 = _mm256_add_pd(x1, x5);
            let s3 = _mm256_add_pd(x3, x7);
            let d1 = _mm256_sub_pd(x1, x5);
            let d3 = _mm256_sub_pd(x3, x7);
            let t3 = cmul_vec64(w2ar, w2ai, s3);
            let p1 = _mm256_add_pd(s1, t3);
            let p5 = _mm256_sub_pd(s1, t3);
            let out_base = dst_base + k;
            let q0 = cmul_vec64(w3ar, w3ai, p1);
            let q2 = cmul_vec64(w3cr, w3ci, p5);
            _mm256_storeu_pd(dst_ptr.add(out_base).cast::<f64>(), _mm256_add_pd(p0, q0));
            _mm256_storeu_pd(
                dst_ptr.add(half_n + out_base).cast::<f64>(),
                _mm256_sub_pd(p0, q0),
            );
            _mm256_storeu_pd(
                dst_ptr.add(quarter_n + out_base).cast::<f64>(),
                _mm256_add_pd(p4, q2),
            );
            _mm256_storeu_pd(
                dst_ptr.add(half_n + quarter_n + out_base).cast::<f64>(),
                _mm256_sub_pd(p4, q2),
            );

            let u2 = cmul_vec64(w2br, w2bi, d2);
            let u3 = cmul_vec64(w2br, w2bi, d3);
            let p2 = _mm256_add_pd(d0, u2);
            let p3 = _mm256_add_pd(d1, u3);
            let p6 = _mm256_sub_pd(d0, u2);
            let p7 = _mm256_sub_pd(d1, u3);
            let q1 = cmul_vec64(w3br, w3bi, p3);
            let q3 = cmul_vec64(w3dr, w3di, p7);
            _mm256_storeu_pd(
                dst_ptr.add(eighth_n + out_base).cast::<f64>(),
                _mm256_add_pd(p2, q1),
            );
            _mm256_storeu_pd(
                dst_ptr.add(half_n + eighth_n + out_base).cast::<f64>(),
                _mm256_sub_pd(p2, q1),
            );
            _mm256_storeu_pd(
                dst_ptr.add(quarter_n + eighth_n + out_base).cast::<f64>(),
                _mm256_add_pd(p6, q3),
            );
            _mm256_storeu_pd(
                dst_ptr
                    .add(half_n + quarter_n + eighth_n + out_base)
                    .cast::<f64>(),
                _mm256_sub_pd(p6, q3),
            );
            k += 2;
        }
    }
}
