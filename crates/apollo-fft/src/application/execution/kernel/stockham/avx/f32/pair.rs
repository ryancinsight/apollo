use super::fixed::cmul_vec32;
use num_complex::Complex32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn stage_pair32_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    let n = src.len();
    let groups = n / (radix << 1);
    let half_groups = groups >> 1;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(groups >= 8);
    debug_assert_eq!(groups & (groups - 1), 0);
    debug_assert_eq!(half_groups & 3, 0);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= radix << 1);

    let vector_end = half_groups;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();

    for j in 0..radix {
        let w1 = *first_ptr.add(j);
        let w2 = *second_ptr.add(j);
        let w3 = *second_ptr.add(j + radix);
        let w1r = _mm256_set1_ps(w1.re);
        let w1i = _mm256_set1_ps(w1.im);
        let w2r = _mm256_set1_ps(w2.re);
        let w2i = _mm256_set1_ps(w2.im);
        let w3r = _mm256_set1_ps(w3.re);
        let w3i = _mm256_set1_ps(w3.im);
        let src_base = j * groups * 2;
        let dst_base = j * half_groups;
        let mut k = 0usize;
        while k < vector_end {
            let x0 = _mm256_loadu_ps(src_ptr.add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(src_ptr.add(src_base + half_groups + k).cast::<f32>());
            let raw_x2 = _mm256_loadu_ps(src_ptr.add(src_base + groups + k).cast::<f32>());
            let raw_x3 = _mm256_loadu_ps(
                src_ptr
                    .add(src_base + groups + half_groups + k)
                    .cast::<f32>(),
            );
            let x2 = cmul_vec32(w1r, w1i, raw_x2);
            let x3 = cmul_vec32(w1r, w1i, raw_x3);
            let a0 = _mm256_add_ps(x0, x2);
            let a1 = _mm256_add_ps(x1, x3);
            let b0 = _mm256_sub_ps(x0, x2);
            let b1 = _mm256_sub_ps(x1, x3);
            let c0 = cmul_vec32(w2r, w2i, a1);
            let c1 = cmul_vec32(w3r, w3i, b1);
            _mm256_storeu_ps(
                dst_ptr.add(dst_base + k).cast::<f32>(),
                _mm256_add_ps(a0, c0),
            );
            _mm256_storeu_ps(
                dst_ptr.add(dst_base + half_n + k).cast::<f32>(),
                _mm256_sub_ps(a0, c0),
            );
            _mm256_storeu_ps(
                dst_ptr.add(dst_base + quarter_n + k).cast::<f32>(),
                _mm256_add_ps(b0, c1),
            );
            _mm256_storeu_ps(
                dst_ptr.add(dst_base + half_n + quarter_n + k).cast::<f32>(),
                _mm256_sub_ps(b0, c1),
            );
            k += 4;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn stage_pair32_radix1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    second_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    let n = src.len();
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(n >= 16);
    debug_assert_eq!(n & (n - 1), 0);
    debug_assert_eq!(quarter_n & 3, 0);
    debug_assert!(second_twiddles.len() >= 2);

    let vector_end = quarter_n;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let w3 = *second_twiddles.as_ptr().add(1);
    let w3r = _mm256_set1_ps(w3.re);
    let w3i = _mm256_set1_ps(w3.im);
    let mut k = 0usize;
    while k < vector_end {
        let x0 = _mm256_loadu_ps(src_ptr.add(k).cast::<f32>());
        let x1 = _mm256_loadu_ps(src_ptr.add(quarter_n + k).cast::<f32>());
        let x2 = _mm256_loadu_ps(src_ptr.add(half_n + k).cast::<f32>());
        let x3 = _mm256_loadu_ps(src_ptr.add(half_n + quarter_n + k).cast::<f32>());
        let a0 = _mm256_add_ps(x0, x2);
        let a1 = _mm256_add_ps(x1, x3);
        let b0 = _mm256_sub_ps(x0, x2);
        let b1 = _mm256_sub_ps(x1, x3);
        let c1 = cmul_vec32(w3r, w3i, b1);
        _mm256_storeu_ps(dst_ptr.add(k).cast::<f32>(), _mm256_add_ps(a0, a1));
        _mm256_storeu_ps(dst_ptr.add(half_n + k).cast::<f32>(), _mm256_sub_ps(a0, a1));
        _mm256_storeu_ps(
            dst_ptr.add(quarter_n + k).cast::<f32>(),
            _mm256_add_ps(b0, c1),
        );
        _mm256_storeu_ps(
            dst_ptr.add(half_n + quarter_n + k).cast::<f32>(),
            _mm256_sub_ps(b0, c1),
        );
        k += 4;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_pair32_quarter_groups_two_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castpd_ps, _mm256_castps256_ps128, _mm256_castps_pd,
        _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_permute2f128_pd, _mm256_set1_ps,
        _mm256_set_ps, _mm256_sub_ps, _mm_storeu_ps,
    };

    let n = src.len();
    let quarter_n = n >> 2;
    let half_n = n >> 1;

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();

    for j in 0..radix {
        let w1 = *first_ptr.add(j);
        let w2 = *second_ptr.add(j);
        let w3 = *second_ptr.add(j + radix);

        let w1r = _mm256_set1_ps(w1.re);
        let w1i = _mm256_set1_ps(w1.im);

        let src_base = j * 8;
        let y01 = _mm256_loadu_ps(src_ptr.add(src_base).cast::<f32>());
        let y23 = _mm256_loadu_ps(src_ptr.add(src_base + 4).cast::<f32>());

        let x23 = cmul_vec32(w1r, w1i, y23);
        let s01 = _mm256_add_ps(y01, x23);
        let d01 = _mm256_sub_ps(y01, x23);

        let s02 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(s01),
            _mm256_castps_pd(d01),
            0x20,
        ));
        let s13 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(s01),
            _mm256_castps_pd(d01),
            0x31,
        ));

        let w23_r = _mm256_set_ps(w3.re, w3.re, w3.re, w3.re, w2.re, w2.re, w2.re, w2.re);
        let w23_i = _mm256_set_ps(w3.im, w3.im, w3.im, w3.im, w2.im, w2.im, w2.im, w2.im);
        let t13 = cmul_vec32(w23_r, w23_i, s13);

        let out02 = _mm256_add_ps(s02, t13);
        let out13 = _mm256_sub_ps(s02, t13);

        let out_base = j * 2;
        _mm_storeu_ps(
            dst_ptr.add(out_base).cast::<f32>(),
            _mm256_castps256_ps128(out02),
        );
        _mm_storeu_ps(
            dst_ptr.add(quarter_n + out_base).cast::<f32>(),
            _mm256_extractf128_ps(out02, 1),
        );
        _mm_storeu_ps(
            dst_ptr.add(half_n + out_base).cast::<f32>(),
            _mm256_castps256_ps128(out13),
        );
        _mm_storeu_ps(
            dst_ptr.add(half_n + quarter_n + out_base).cast::<f32>(),
            _mm256_extractf128_ps(out13, 1),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn stage_pair32_groups_two_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castpd_ps, _mm256_castps_pd, _mm256_loadu_ps, _mm256_movehdup_ps,
        _mm256_moveldup_ps, _mm256_permute2f128_pd, _mm256_storeu_ps, _mm256_sub_ps,
        _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };

    let n = src.len();
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, radix << 2);
    debug_assert_eq!(radix & (radix - 1), 0);
    debug_assert_eq!(radix & 3, 0);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= radix << 1);

    let vector_end = radix;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();

    let mut j = 0usize;
    while j < vector_end {
        let d0 = _mm256_loadu_ps(src_ptr.add(j * 4).cast::<f32>());
        let d1 = _mm256_loadu_ps(src_ptr.add((j + 1) * 4).cast::<f32>());
        let d2 = _mm256_loadu_ps(src_ptr.add((j + 2) * 4).cast::<f32>());
        let d3 = _mm256_loadu_ps(src_ptr.add((j + 3) * 4).cast::<f32>());

        let t0 = _mm256_castpd_ps(_mm256_unpacklo_pd(
            _mm256_castps_pd(d0),
            _mm256_castps_pd(d1),
        ));
        let t1 = _mm256_castpd_ps(_mm256_unpackhi_pd(
            _mm256_castps_pd(d0),
            _mm256_castps_pd(d1),
        ));
        let t2 = _mm256_castpd_ps(_mm256_unpacklo_pd(
            _mm256_castps_pd(d2),
            _mm256_castps_pd(d3),
        ));
        let t3 = _mm256_castpd_ps(_mm256_unpackhi_pd(
            _mm256_castps_pd(d2),
            _mm256_castps_pd(d3),
        ));

        let x0 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(t0),
            _mm256_castps_pd(t2),
            0x20,
        ));
        let x2 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(t0),
            _mm256_castps_pd(t2),
            0x31,
        ));
        let x1 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(t1),
            _mm256_castps_pd(t3),
            0x20,
        ));
        let x3 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(t1),
            _mm256_castps_pd(t3),
            0x31,
        ));

        let w1 = _mm256_loadu_ps(first_ptr.add(j).cast::<f32>());
        let w1r = _mm256_moveldup_ps(w1);
        let w1i = _mm256_movehdup_ps(w1);

        let x2_mul = cmul_vec32(w1r, w1i, x2);
        let x3_mul = cmul_vec32(w1r, w1i, x3);

        let a0 = _mm256_add_ps(x0, x2_mul);
        let a1 = _mm256_add_ps(x1, x3_mul);
        let b0 = _mm256_sub_ps(x0, x2_mul);
        let b1 = _mm256_sub_ps(x1, x3_mul);

        let w2 = _mm256_loadu_ps(second_ptr.add(j).cast::<f32>());
        let w3 = _mm256_loadu_ps(second_ptr.add(j + radix).cast::<f32>());

        let w2r = _mm256_moveldup_ps(w2);
        let w2i = _mm256_movehdup_ps(w2);
        let c0 = cmul_vec32(w2r, w2i, a1);

        let w3r = _mm256_moveldup_ps(w3);
        let w3i = _mm256_movehdup_ps(w3);
        let c1 = cmul_vec32(w3r, w3i, b1);

        _mm256_storeu_ps(dst_ptr.add(j).cast::<f32>(), _mm256_add_ps(a0, c0));
        _mm256_storeu_ps(dst_ptr.add(j + half_n).cast::<f32>(), _mm256_sub_ps(a0, c0));
        _mm256_storeu_ps(
            dst_ptr.add(j + quarter_n).cast::<f32>(),
            _mm256_add_ps(b0, c1),
        );
        _mm256_storeu_ps(
            dst_ptr.add(j + half_n + quarter_n).cast::<f32>(),
            _mm256_sub_ps(b0, c1),
        );

        j += 4;
    }
}
