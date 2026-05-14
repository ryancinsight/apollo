use num_complex::Complex64;
use super::fixed::cmul_vec64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage64_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    radix: usize,
    twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd,
        _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };

    let n = src.len();
    let half_n = n >> 1;
    let groups = n / (radix << 1);
    let vector_end = groups & !1usize;
    for j in 0..radix {
        let w = twiddles[j];
        let wr = _mm256_set1_pd(w.re);
        let wi = _mm256_set1_pd(w.im);
        let src_base = j * groups * 2;
        let dst_base = j * groups;
        let mut k = 0usize;
        while k < vector_end {
            let a = _mm256_loadu_pd(src.as_ptr().add(src_base + k).cast::<f64>());
            let b = _mm256_loadu_pd(src.as_ptr().add(src_base + groups + k).cast::<f64>());
            let swapped = _mm256_permute_pd::<0b0101>(b);
            let product = _mm256_fmaddsub_pd(wr, b, _mm256_mul_pd(wi, swapped));
            _mm256_storeu_pd(
                dst.as_mut_ptr().add(dst_base + k).cast::<f64>(),
                _mm256_add_pd(a, product),
            );
            _mm256_storeu_pd(
                dst.as_mut_ptr().add(dst_base + half_n + k).cast::<f64>(),
                _mm256_sub_pd(a, product),
            );
            k += 2;
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

/// AVX/FMA final Stockham f64 stage for `groups == 1`.
///
/// For `N = 2R`, the single remaining Stockham stage is
///
/// `dst[j]     = src[2j] + W_N^j src[2j+1]`
/// `dst[R + j] = src[2j] - W_N^j src[2j+1]`.
///
/// The leaf packs two adjacent `j` values as
/// `[src[2j], src[2j+2]]` and `[src[2j+1], src[2j+3]]` in separate YMM
/// registers, then applies the twiddle vector
/// `[W_N^j, W_N^(j+1)]`. This is the same DAG as the scalar recurrence with
/// only a representation change; no cross-lane FFT dependency is introduced.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage64_groups_one_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    radix: usize,
    twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_permute2f128_pd, _mm256_permute_pd,
        _mm256_storeu_pd, _mm256_sub_pd,
    };

    debug_assert_eq!(src.len(), radix << 1);
    debug_assert_eq!(dst.len(), src.len());
    debug_assert!(radix >= 2);
    debug_assert_eq!(radix & 1, 0);
    debug_assert!(twiddles.len() >= radix);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let twiddle_ptr = twiddles.as_ptr();
    let half_n = radix;
    let mut j = 0usize;
    while j < radix {
        let x01 = _mm256_loadu_pd(src_ptr.add(j << 1).cast::<f64>());
        let x23 = _mm256_loadu_pd(src_ptr.add((j << 1) + 2).cast::<f64>());
        let a = _mm256_permute2f128_pd(x01, x23, 0x20);
        let b = _mm256_permute2f128_pd(x01, x23, 0x31);
        let w = _mm256_loadu_pd(twiddle_ptr.add(j).cast::<f64>());
        let wr = _mm256_permute_pd::<0b0000>(w);
        let wi = _mm256_permute_pd::<0b1111>(w);
        let product = cmul_vec64(wr, wi, b);

        _mm256_storeu_pd(dst_ptr.add(j).cast::<f64>(), _mm256_add_pd(a, product));
        _mm256_storeu_pd(
            dst_ptr.add(half_n + j).cast::<f64>(),
            _mm256_sub_pd(a, product),
        );
        j += 2;
    }
}
