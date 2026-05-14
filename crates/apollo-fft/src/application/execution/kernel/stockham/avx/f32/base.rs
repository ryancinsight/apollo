use num_complex::Complex32;
use super::fixed::{cmul_vec32, cmul_pair32};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn stage32_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps,
        _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    let n = src.len();
    let half_n = n >> 1;
    let groups = n / (radix << 1);
    let vector_end = groups & !3usize;
    for j in 0..radix {
        let w = twiddles[j];
        let wr = _mm256_set1_ps(w.re);
        let wi = _mm256_set1_ps(w.im);
        let src_base = j * groups * 2;
        let dst_base = j * groups;
        let mut k = 0usize;
        while k < vector_end {
            let a = _mm256_loadu_ps(src.as_ptr().add(src_base + k).cast::<f32>());
            let b = _mm256_loadu_ps(src.as_ptr().add(src_base + groups + k).cast::<f32>());
            let swapped = _mm256_permute_ps::<0b1011_0001>(b);
            let product = _mm256_fmaddsub_ps(wr, b, _mm256_mul_ps(wi, swapped));
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(dst_base + k).cast::<f32>(),
                _mm256_add_ps(a, product),
            );
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(dst_base + half_n + k).cast::<f32>(),
                _mm256_sub_ps(a, product),
            );
            k += 4;
        }
        while k < groups {
            let a = src[src_base + k];
            let b = src[src_base + groups + k] * w;
            dst[dst_base + k] = a + b;
            dst[dst_base + half_n + k] = a - b;
            k += 1;
        }
    }
}

/// AVX/FMA Stockham f32 stage over two independent complex instances per vector
/// for `groups == 1`.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn stage32_groups_one_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm_add_ps, _mm_castpd_ps, _mm_castps_pd, _mm_loadu_ps, _mm_movehdup_ps, _mm_moveldup_ps,
        _mm_storeu_ps, _mm_sub_ps, _mm_unpackhi_pd, _mm_unpacklo_pd,
    };

    let n = src.len();
    let half_n = n >> 1;
    debug_assert_eq!(n, radix << 1);
    debug_assert!(radix >= 2);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let twiddle_ptr = twiddles.as_ptr();
    let vector_end = radix & !1usize;
    let mut j = 0usize;
    while j < vector_end {
        let x0 = _mm_loadu_ps(src_ptr.add(j << 1).cast::<f32>());
        let x1 = _mm_loadu_ps(src_ptr.add((j << 1) + 2).cast::<f32>());

        let a = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(x0), _mm_castps_pd(x1)));
        let b = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(x0), _mm_castps_pd(x1)));

        let w = _mm_loadu_ps(twiddle_ptr.add(j).cast::<f32>());
        let wr = _mm_moveldup_ps(w);
        let wi = _mm_movehdup_ps(w);
        let product = cmul_pair32(wr, wi, b);

        _mm_storeu_ps(dst_ptr.add(j).cast::<f32>(), _mm_add_ps(a, product));
        _mm_storeu_ps(
            dst_ptr.add(half_n + j).cast::<f32>(),
            _mm_sub_ps(a, product),
        );
        j += 2;
    }
    while j < radix {
        let a = src[j << 1];
        let b = src[(j << 1) + 1] * twiddles[j];
        dst[j] = a + b;
        dst[half_n + j] = a - b;
        j += 1;
    }
}
