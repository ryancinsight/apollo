use super::fixed::cmul_pair32;
use num_complex::Complex32;

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
