use super::fixed::cmul_vec_precise;
use num_complex::Complex64;

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
        let product = cmul_vec_precise(wr, wi, b);

        _mm256_storeu_pd(dst_ptr.add(j).cast::<f64>(), _mm256_add_pd(a, product));
        _mm256_storeu_pd(
            dst_ptr.add(half_n + j).cast::<f64>(),
            _mm256_sub_pd(a, product),
        );
        j += 2;
    }
}
