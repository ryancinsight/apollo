use super::fixed::cmul_vec_reduced;
use eunomia::Complex32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_pair_quarter_groups_two_reduced_avx_fma(
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

        let x23 = cmul_vec_reduced(w1r, w1i, y23);
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
        let t13 = cmul_vec_reduced(w23_r, w23_i, s13);

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
