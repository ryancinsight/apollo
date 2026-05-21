use super::fixed::cmul_vec64;
use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple64_groups_eight_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    radix: usize,
    first_twiddles: &[Complex64],
    second_twiddles: &[Complex64],
    third_twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };

    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, radix << 4);
    debug_assert_eq!(dst.len(), n);
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
        let w1r = _mm256_set1_pd(w1.re);
        let w1i = _mm256_set1_pd(w1.im);
        let src_base = j << 4;
        let out_base = j << 1;

        let x0 = _mm256_loadu_pd(src_ptr.add(src_base).cast::<f64>());
        let x2 = _mm256_loadu_pd(src_ptr.add(src_base + 4).cast::<f64>());
        let x4 = cmul_vec64(
            w1r,
            w1i,
            _mm256_loadu_pd(src_ptr.add(src_base + 8).cast::<f64>()),
        );
        let x6 = cmul_vec64(
            w1r,
            w1i,
            _mm256_loadu_pd(src_ptr.add(src_base + 12).cast::<f64>()),
        );

        let s0 = _mm256_add_pd(x0, x4);
        let s2 = _mm256_add_pd(x2, x6);
        let d0 = _mm256_sub_pd(x0, x4);
        let d2 = _mm256_sub_pd(x2, x6);

        let w2a = *second_ptr.add(j);
        let t2 = cmul_vec64(_mm256_set1_pd(w2a.re), _mm256_set1_pd(w2a.im), s2);
        let p0 = _mm256_add_pd(s0, t2);
        let p4 = _mm256_sub_pd(s0, t2);

        let x1 = _mm256_loadu_pd(src_ptr.add(src_base + 2).cast::<f64>());
        let x3 = _mm256_loadu_pd(src_ptr.add(src_base + 6).cast::<f64>());
        let x5 = cmul_vec64(
            w1r,
            w1i,
            _mm256_loadu_pd(src_ptr.add(src_base + 10).cast::<f64>()),
        );
        let x7 = cmul_vec64(
            w1r,
            w1i,
            _mm256_loadu_pd(src_ptr.add(src_base + 14).cast::<f64>()),
        );

        let s1 = _mm256_add_pd(x1, x5);
        let s3 = _mm256_add_pd(x3, x7);
        let d1 = _mm256_sub_pd(x1, x5);
        let d3 = _mm256_sub_pd(x3, x7);
        let t3 = cmul_vec64(_mm256_set1_pd(w2a.re), _mm256_set1_pd(w2a.im), s3);
        let p1 = _mm256_add_pd(s1, t3);
        let p5 = _mm256_sub_pd(s1, t3);

        let w3a = *third_ptr.add(j);
        let q0 = cmul_vec64(_mm256_set1_pd(w3a.re), _mm256_set1_pd(w3a.im), p1);
        _mm256_storeu_pd(dst_ptr.add(out_base).cast::<f64>(), _mm256_add_pd(p0, q0));
        _mm256_storeu_pd(
            dst_ptr.add(half_n + out_base).cast::<f64>(),
            _mm256_sub_pd(p0, q0),
        );

        let w3c = *third_ptr.add(j + 2 * radix);
        let q2 = cmul_vec64(_mm256_set1_pd(w3c.re), _mm256_set1_pd(w3c.im), p5);
        _mm256_storeu_pd(
            dst_ptr.add(quarter_n + out_base).cast::<f64>(),
            _mm256_add_pd(p4, q2),
        );
        _mm256_storeu_pd(
            dst_ptr.add(half_n + quarter_n + out_base).cast::<f64>(),
            _mm256_sub_pd(p4, q2),
        );

        let w2b = *second_ptr.add(j + radix);
        let w2br = _mm256_set1_pd(w2b.re);
        let w2bi = _mm256_set1_pd(w2b.im);
        let u2 = cmul_vec64(w2br, w2bi, d2);
        let u3 = cmul_vec64(w2br, w2bi, d3);
        let p2 = _mm256_add_pd(d0, u2);
        let p3 = _mm256_add_pd(d1, u3);
        let p6 = _mm256_sub_pd(d0, u2);
        let p7 = _mm256_sub_pd(d1, u3);

        let w3b = *third_ptr.add(j + radix);
        let q1 = cmul_vec64(_mm256_set1_pd(w3b.re), _mm256_set1_pd(w3b.im), p3);
        _mm256_storeu_pd(
            dst_ptr.add(eighth_n + out_base).cast::<f64>(),
            _mm256_add_pd(p2, q1),
        );
        _mm256_storeu_pd(
            dst_ptr.add(half_n + eighth_n + out_base).cast::<f64>(),
            _mm256_sub_pd(p2, q1),
        );

        let w3d = *third_ptr.add(j + 3 * radix);
        let q3 = cmul_vec64(_mm256_set1_pd(w3d.re), _mm256_set1_pd(w3d.im), p7);
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
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple64_throughput_avx_fma(
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
            let x1 = _mm256_loadu_pd(src_ptr.add(src_base + quarter_groups + k).cast::<f64>());
            let x2 = _mm256_loadu_pd(src_ptr.add(src_base + 2 * quarter_groups + k).cast::<f64>());
            let x3 = _mm256_loadu_pd(src_ptr.add(src_base + 3 * quarter_groups + k).cast::<f64>());
            let x4 = cmul_vec64(
                w1r,
                w1i,
                _mm256_loadu_pd(src_ptr.add(src_base + groups + k).cast::<f64>()),
            );
            let x5 = cmul_vec64(
                w1r,
                w1i,
                _mm256_loadu_pd(
                    src_ptr
                        .add(src_base + groups + quarter_groups + k)
                        .cast::<f64>(),
                ),
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
            let x7 = cmul_vec64(
                w1r,
                w1i,
                _mm256_loadu_pd(
                    src_ptr
                        .add(src_base + groups + 3 * quarter_groups + k)
                        .cast::<f64>(),
                ),
            );

            let s0 = _mm256_add_pd(x0, x4);
            let s1 = _mm256_add_pd(x1, x5);
            let s2 = _mm256_add_pd(x2, x6);
            let s3 = _mm256_add_pd(x3, x7);
            let d0 = _mm256_sub_pd(x0, x4);
            let d1 = _mm256_sub_pd(x1, x5);
            let d2 = _mm256_sub_pd(x2, x6);
            let d3 = _mm256_sub_pd(x3, x7);

            let t2 = cmul_vec64(w2ar, w2ai, s2);
            let t3 = cmul_vec64(w2ar, w2ai, s3);
            let u2 = cmul_vec64(w2br, w2bi, d2);
            let u3 = cmul_vec64(w2br, w2bi, d3);
            let p0 = _mm256_add_pd(s0, t2);
            let p1 = _mm256_add_pd(s1, t3);
            let p2 = _mm256_add_pd(d0, u2);
            let p3 = _mm256_add_pd(d1, u3);
            let p4 = _mm256_sub_pd(s0, t2);
            let p5 = _mm256_sub_pd(s1, t3);
            let p6 = _mm256_sub_pd(d0, u2);
            let p7 = _mm256_sub_pd(d1, u3);

            let q0 = cmul_vec64(w3ar, w3ai, p1);
            let q1 = cmul_vec64(w3br, w3bi, p3);
            let q2 = cmul_vec64(w3cr, w3ci, p5);
            let q3 = cmul_vec64(w3dr, w3di, p7);
            let out_base = dst_base + k;
            _mm256_storeu_pd(dst_ptr.add(out_base).cast::<f64>(), _mm256_add_pd(p0, q0));
            _mm256_storeu_pd(
                dst_ptr.add(half_n + out_base).cast::<f64>(),
                _mm256_sub_pd(p0, q0),
            );
            _mm256_storeu_pd(
                dst_ptr.add(eighth_n + out_base).cast::<f64>(),
                _mm256_add_pd(p2, q1),
            );
            _mm256_storeu_pd(
                dst_ptr.add(half_n + eighth_n + out_base).cast::<f64>(),
                _mm256_sub_pd(p2, q1),
            );
            _mm256_storeu_pd(
                dst_ptr.add(quarter_n + out_base).cast::<f64>(),
                _mm256_add_pd(p4, q2),
            );
            _mm256_storeu_pd(
                dst_ptr.add(half_n + quarter_n + out_base).cast::<f64>(),
                _mm256_sub_pd(p4, q2),
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
