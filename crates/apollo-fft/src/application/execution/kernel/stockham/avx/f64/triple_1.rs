use super::fixed::cmul_vec64;
use num_complex::Complex64;

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
