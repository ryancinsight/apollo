use super::fixed::{avx_rotate_quarter_turn32, cmul_vec32};
use num_complex::Complex32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple32_radix1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    second_twiddles: &[Complex32],
    third_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_set_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };

    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(n >= 32);
    debug_assert_eq!(n & (n - 1), 0);
    debug_assert!(second_twiddles.len() >= 2);
    debug_assert!(third_twiddles.len() >= 4);
    debug_assert_eq!(eighth_n & 3, 0);

    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();
    let w2b = *second_ptr.add(1);
    let w3b = *third_ptr.add(1);
    let w3c = *third_ptr.add(2);
    let w3d = *third_ptr.add(3);
    let w3br = _mm256_set1_ps(w3b.re);
    let w3bi = _mm256_set1_ps(w3b.im);
    let w3dr = _mm256_set1_ps(w3d.re);
    let w3di = _mm256_set1_ps(w3d.im);
    let w2_quarter_turn_mask = if w2b.im > 0.0 {
        _mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0)
    } else {
        _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0)
    };
    let w3_quarter_turn_mask = if w3c.im > 0.0 {
        _mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0)
    } else {
        _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0)
    };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mut k = 0usize;
    while k < eighth_n {
        let x0 = _mm256_loadu_ps(src_ptr.add(k).cast::<f32>());
        let x2 = _mm256_loadu_ps(src_ptr.add(2 * eighth_n + k).cast::<f32>());
        let x4 = _mm256_loadu_ps(src_ptr.add(4 * eighth_n + k).cast::<f32>());
        let x6 = _mm256_loadu_ps(src_ptr.add(6 * eighth_n + k).cast::<f32>());

        let s0 = _mm256_add_ps(x0, x4);
        let s2 = _mm256_add_ps(x2, x6);
        let d0 = _mm256_sub_ps(x0, x4);
        let d2 = _mm256_sub_ps(x2, x6);

        let u2 = avx_rotate_quarter_turn32(d2, w2_quarter_turn_mask);
        let p0 = _mm256_add_ps(s0, s2);
        let p2 = _mm256_add_ps(d0, u2);
        let p4 = _mm256_sub_ps(s0, s2);
        let p6 = _mm256_sub_ps(d0, u2);

        let x1 = _mm256_loadu_ps(src_ptr.add(eighth_n + k).cast::<f32>());
        let x3 = _mm256_loadu_ps(src_ptr.add(3 * eighth_n + k).cast::<f32>());
        let x5 = _mm256_loadu_ps(src_ptr.add(5 * eighth_n + k).cast::<f32>());
        let x7 = _mm256_loadu_ps(src_ptr.add(7 * eighth_n + k).cast::<f32>());

        let s1 = _mm256_add_ps(x1, x5);
        let s3 = _mm256_add_ps(x3, x7);
        let d1 = _mm256_sub_ps(x1, x5);
        let d3 = _mm256_sub_ps(x3, x7);
        let u3 = avx_rotate_quarter_turn32(d3, w2_quarter_turn_mask);
        let p1 = _mm256_add_ps(s1, s3);
        let p3 = _mm256_add_ps(d1, u3);
        let p5 = _mm256_sub_ps(s1, s3);
        let p7 = _mm256_sub_ps(d1, u3);

        let q2 = avx_rotate_quarter_turn32(p5, w3_quarter_turn_mask);
        _mm256_storeu_ps(dst_ptr.add(k).cast::<f32>(), _mm256_add_ps(p0, p1));
        _mm256_storeu_ps(dst_ptr.add(half_n + k).cast::<f32>(), _mm256_sub_ps(p0, p1));
        _mm256_storeu_ps(
            dst_ptr.add(quarter_n + k).cast::<f32>(),
            _mm256_add_ps(p4, q2),
        );
        _mm256_storeu_ps(
            dst_ptr.add(half_n + quarter_n + k).cast::<f32>(),
            _mm256_sub_ps(p4, q2),
        );

        let q1 = cmul_vec32(w3br, w3bi, p3);
        let q3 = cmul_vec32(w3dr, w3di, p7);
        _mm256_storeu_ps(
            dst_ptr.add(eighth_n + k).cast::<f32>(),
            _mm256_add_ps(p2, q1),
        );
        _mm256_storeu_ps(
            dst_ptr.add(half_n + eighth_n + k).cast::<f32>(),
            _mm256_sub_ps(p2, q1),
        );
        _mm256_storeu_ps(
            dst_ptr.add(quarter_n + eighth_n + k).cast::<f32>(),
            _mm256_add_ps(p6, q3),
        );
        _mm256_storeu_ps(
            dst_ptr.add(half_n + quarter_n + eighth_n + k).cast::<f32>(),
            _mm256_sub_ps(p6, q3),
        );
        k += 4;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple32_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    groups: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
    third_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    let n = src.len();
    let quarter_groups = groups >> 2;
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(groups >= 8);
    debug_assert_eq!(groups & (groups - 1), 0);
    debug_assert_eq!(quarter_groups & 3, 0);
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
        let w1r = _mm256_set1_ps(w1.re);
        let w1i = _mm256_set1_ps(w1.im);
        let w2ar = _mm256_set1_ps(w2a.re);
        let w2ai = _mm256_set1_ps(w2a.im);
        let src_base = j * groups * 2;
        let dst_base = j * quarter_groups;
        let mut k = 0usize;
        while k < quarter_groups {
            let x0 = _mm256_loadu_ps(src_ptr.add(src_base + k).cast::<f32>());
            let x2 = _mm256_loadu_ps(src_ptr.add(src_base + 2 * quarter_groups + k).cast::<f32>());
            let x4 = cmul_vec32(
                w1r,
                w1i,
                _mm256_loadu_ps(src_ptr.add(src_base + groups + k).cast::<f32>()),
            );
            let x6 = cmul_vec32(
                w1r,
                w1i,
                _mm256_loadu_ps(
                    src_ptr
                        .add(src_base + groups + 2 * quarter_groups + k)
                        .cast::<f32>(),
                ),
            );

            let s0 = _mm256_add_ps(x0, x4);
            let s2 = _mm256_add_ps(x2, x6);
            let d0 = _mm256_sub_ps(x0, x4);
            let d2 = _mm256_sub_ps(x2, x6);

            let t2 = cmul_vec32(w2ar, w2ai, s2);
            let p0 = _mm256_add_ps(s0, t2);
            let p4 = _mm256_sub_ps(s0, t2);

            let x1 = _mm256_loadu_ps(src_ptr.add(src_base + quarter_groups + k).cast::<f32>());
            let x3 = _mm256_loadu_ps(src_ptr.add(src_base + 3 * quarter_groups + k).cast::<f32>());
            let x5 = cmul_vec32(
                w1r,
                w1i,
                _mm256_loadu_ps(
                    src_ptr
                        .add(src_base + groups + quarter_groups + k)
                        .cast::<f32>(),
                ),
            );
            let x7 = cmul_vec32(
                w1r,
                w1i,
                _mm256_loadu_ps(
                    src_ptr
                        .add(src_base + groups + 3 * quarter_groups + k)
                        .cast::<f32>(),
                ),
            );

            let s1 = _mm256_add_ps(x1, x5);
            let s3 = _mm256_add_ps(x3, x7);
            let d1 = _mm256_sub_ps(x1, x5);
            let d3 = _mm256_sub_ps(x3, x7);

            let t3 = cmul_vec32(w2ar, w2ai, s3);
            let p1 = _mm256_add_ps(s1, t3);
            let p5 = _mm256_sub_ps(s1, t3);

            let out_base = dst_base + k;
            let q0 = cmul_vec32(_mm256_set1_ps(w3a.re), _mm256_set1_ps(w3a.im), p1);
            let q2 = cmul_vec32(_mm256_set1_ps(w3c.re), _mm256_set1_ps(w3c.im), p5);
            _mm256_storeu_ps(dst_ptr.add(out_base).cast::<f32>(), _mm256_add_ps(p0, q0));
            _mm256_storeu_ps(
                dst_ptr.add(half_n + out_base).cast::<f32>(),
                _mm256_sub_ps(p0, q0),
            );
            _mm256_storeu_ps(
                dst_ptr.add(quarter_n + out_base).cast::<f32>(),
                _mm256_add_ps(p4, q2),
            );
            _mm256_storeu_ps(
                dst_ptr.add(half_n + quarter_n + out_base).cast::<f32>(),
                _mm256_sub_ps(p4, q2),
            );

            let w2br = _mm256_set1_ps(w2b.re);
            let w2bi = _mm256_set1_ps(w2b.im);
            let u2 = cmul_vec32(w2br, w2bi, d2);
            let u3 = cmul_vec32(w2br, w2bi, d3);
            let p2 = _mm256_add_ps(d0, u2);
            let p3 = _mm256_add_ps(d1, u3);
            let p6 = _mm256_sub_ps(d0, u2);
            let p7 = _mm256_sub_ps(d1, u3);
            let q1 = cmul_vec32(_mm256_set1_ps(w3b.re), _mm256_set1_ps(w3b.im), p3);
            let q3 = cmul_vec32(_mm256_set1_ps(w3d.re), _mm256_set1_ps(w3d.im), p7);
            _mm256_storeu_ps(
                dst_ptr.add(eighth_n + out_base).cast::<f32>(),
                _mm256_add_ps(p2, q1),
            );
            _mm256_storeu_ps(
                dst_ptr.add(half_n + eighth_n + out_base).cast::<f32>(),
                _mm256_sub_ps(p2, q1),
            );
            _mm256_storeu_ps(
                dst_ptr.add(quarter_n + eighth_n + out_base).cast::<f32>(),
                _mm256_add_ps(p6, q3),
            );
            _mm256_storeu_ps(
                dst_ptr
                    .add(half_n + quarter_n + eighth_n + out_base)
                    .cast::<f32>(),
                _mm256_sub_ps(p6, q3),
            );
            k += 4;
        }
    }
}

/// AVX/FMA f32 three-stage fused Stockham radix-8 kernel (low register-pressure variant).
///
/// Structurally identical to `stage_triple64_low_live_avx_fma` with f32 YMM arithmetic:
/// each YMM holds four `Complex32` values (8 f32 = 256 bits). Within the k-loop the
/// even-index source values (`x0, x2, x4, x6`) are loaded, butterflied, and partially
/// stored before the odd-index values (`x1, x3, x5, x7`) are loaded, reducing peak
/// live registers from ~18 (in `stage_triple32_avx_fma`) to ~14, eliminating register
/// spills and shrinking the stack frame for L1-resident shapes.
///
/// Dispatch routes `n <= 2048` Complex32 (L1-resident) here and larger shapes to
/// `stage_triple32_avx_fma`.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_triple32_low_live_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    groups: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
    third_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    let n = src.len();
    let quarter_groups = groups >> 2;
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(groups >= 8);
    debug_assert_eq!(groups & (groups - 1), 0);
    debug_assert_eq!(quarter_groups & 3, 0);
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
        let w1r = _mm256_set1_ps(w1.re);
        let w1i = _mm256_set1_ps(w1.im);
        let w2ar = _mm256_set1_ps(w2a.re);
        let w2ai = _mm256_set1_ps(w2a.im);
        let w2br = _mm256_set1_ps(w2b.re);
        let w2bi = _mm256_set1_ps(w2b.im);
        let w3ar = _mm256_set1_ps(w3a.re);
        let w3ai = _mm256_set1_ps(w3a.im);
        let w3br = _mm256_set1_ps(w3b.re);
        let w3bi = _mm256_set1_ps(w3b.im);
        let w3cr = _mm256_set1_ps(w3c.re);
        let w3ci = _mm256_set1_ps(w3c.im);
        let w3dr = _mm256_set1_ps(w3d.re);
        let w3di = _mm256_set1_ps(w3d.im);
        let src_base = j * groups * 2;
        let dst_base = j * quarter_groups;
        let mut k = 0usize;
        while k < quarter_groups {
            // Even-index sources: x0, x2, x4, x6 → s0, s2, d0, d2 → p0, p4.
            let x0 = _mm256_loadu_ps(src_ptr.add(src_base + k).cast::<f32>());
            let x2 = _mm256_loadu_ps(src_ptr.add(src_base + 2 * quarter_groups + k).cast::<f32>());
            let x4 = cmul_vec32(
                w1r,
                w1i,
                _mm256_loadu_ps(src_ptr.add(src_base + groups + k).cast::<f32>()),
            );
            let x6 = cmul_vec32(
                w1r,
                w1i,
                _mm256_loadu_ps(
                    src_ptr
                        .add(src_base + groups + 2 * quarter_groups + k)
                        .cast::<f32>(),
                ),
            );

            let s0 = _mm256_add_ps(x0, x4);
            let s2 = _mm256_add_ps(x2, x6);
            let d0 = _mm256_sub_ps(x0, x4);
            let d2 = _mm256_sub_ps(x2, x6);

            let t2 = cmul_vec32(w2ar, w2ai, s2);
            let p0 = _mm256_add_ps(s0, t2);
            let p4 = _mm256_sub_ps(s0, t2);

            // Odd-index sources: x1, x3, x5, x7 → s1, s3, d1, d3 → p1, p5.
            let x1 = _mm256_loadu_ps(src_ptr.add(src_base + quarter_groups + k).cast::<f32>());
            let x3 = _mm256_loadu_ps(src_ptr.add(src_base + 3 * quarter_groups + k).cast::<f32>());
            let x5 = cmul_vec32(
                w1r,
                w1i,
                _mm256_loadu_ps(
                    src_ptr
                        .add(src_base + groups + quarter_groups + k)
                        .cast::<f32>(),
                ),
            );
            let x7 = cmul_vec32(
                w1r,
                w1i,
                _mm256_loadu_ps(
                    src_ptr
                        .add(src_base + groups + 3 * quarter_groups + k)
                        .cast::<f32>(),
                ),
            );
            let s1 = _mm256_add_ps(x1, x5);
            let s3 = _mm256_add_ps(x3, x7);
            let d1 = _mm256_sub_ps(x1, x5);
            let d3 = _mm256_sub_ps(x3, x7);

            let t3 = cmul_vec32(w2ar, w2ai, s3);
            let p1 = _mm256_add_ps(s1, t3);
            let p5 = _mm256_sub_ps(s1, t3);

            // Store s-side outputs (p0, p4): p0 and p4 are dead after this, freeing registers
            // for the d-side computation before p2/p3/p6/p7 are in flight simultaneously.
            let out_base = dst_base + k;
            let q0 = cmul_vec32(w3ar, w3ai, p1);
            let q2 = cmul_vec32(w3cr, w3ci, p5);
            _mm256_storeu_ps(dst_ptr.add(out_base).cast::<f32>(), _mm256_add_ps(p0, q0));
            _mm256_storeu_ps(
                dst_ptr.add(half_n + out_base).cast::<f32>(),
                _mm256_sub_ps(p0, q0),
            );
            _mm256_storeu_ps(
                dst_ptr.add(quarter_n + out_base).cast::<f32>(),
                _mm256_add_ps(p4, q2),
            );
            _mm256_storeu_ps(
                dst_ptr.add(half_n + quarter_n + out_base).cast::<f32>(),
                _mm256_sub_ps(p4, q2),
            );

            // D-side outputs.
            let u2 = cmul_vec32(w2br, w2bi, d2);
            let u3 = cmul_vec32(w2br, w2bi, d3);
            let p2 = _mm256_add_ps(d0, u2);
            let p3 = _mm256_add_ps(d1, u3);
            let p6 = _mm256_sub_ps(d0, u2);
            let p7 = _mm256_sub_ps(d1, u3);
            let q1 = cmul_vec32(w3br, w3bi, p3);
            let q3 = cmul_vec32(w3dr, w3di, p7);
            _mm256_storeu_ps(
                dst_ptr.add(eighth_n + out_base).cast::<f32>(),
                _mm256_add_ps(p2, q1),
            );
            _mm256_storeu_ps(
                dst_ptr.add(half_n + eighth_n + out_base).cast::<f32>(),
                _mm256_sub_ps(p2, q1),
            );
            _mm256_storeu_ps(
                dst_ptr.add(quarter_n + eighth_n + out_base).cast::<f32>(),
                _mm256_add_ps(p6, q3),
            );
            _mm256_storeu_ps(
                dst_ptr
                    .add(half_n + quarter_n + eighth_n + out_base)
                    .cast::<f32>(),
                _mm256_sub_ps(p6, q3),
            );
            k += 4;
        }
    }
}
