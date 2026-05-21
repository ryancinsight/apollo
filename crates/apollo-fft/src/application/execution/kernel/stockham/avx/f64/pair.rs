use super::fixed::cmul_vec64;
use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_pair64_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    radix: usize,
    first_twiddles: &[Complex64],
    second_twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };

    let n = src.len();
    let groups = n / (radix << 1);
    let half_groups = groups >> 1;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    let vector_end = half_groups & !1usize;

    for j in 0..radix {
        let w1 = first_twiddles[j];
        let w2 = second_twiddles[j];
        let w3 = second_twiddles[j + radix];
        let w1r = _mm256_set1_pd(w1.re);
        let w1i = _mm256_set1_pd(w1.im);
        let w2r = _mm256_set1_pd(w2.re);
        let w2i = _mm256_set1_pd(w2.im);
        let w3r = _mm256_set1_pd(w3.re);
        let w3i = _mm256_set1_pd(w3.im);
        let src_base = j * groups * 2;
        let dst_base = j * half_groups;
        let mut k = 0usize;
        while k < vector_end {
            let x0 = _mm256_loadu_pd(src.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(src.as_ptr().add(src_base + half_groups + k).cast::<f64>());
            let raw_x2 = _mm256_loadu_pd(src.as_ptr().add(src_base + groups + k).cast::<f64>());
            let raw_x3 = _mm256_loadu_pd(
                src.as_ptr()
                    .add(src_base + groups + half_groups + k)
                    .cast::<f64>(),
            );
            let x2 = cmul_vec64(w1r, w1i, raw_x2);
            let x3 = cmul_vec64(w1r, w1i, raw_x3);
            let a0 = _mm256_add_pd(x0, x2);
            let a1 = _mm256_add_pd(x1, x3);
            let b0 = _mm256_sub_pd(x0, x2);
            let b1 = _mm256_sub_pd(x1, x3);
            let c0 = cmul_vec64(w2r, w2i, a1);
            let c1 = cmul_vec64(w3r, w3i, b1);
            _mm256_storeu_pd(
                dst.as_mut_ptr().add(dst_base + k).cast::<f64>(),
                _mm256_add_pd(a0, c0),
            );
            _mm256_storeu_pd(
                dst.as_mut_ptr().add(dst_base + half_n + k).cast::<f64>(),
                _mm256_sub_pd(a0, c0),
            );
            _mm256_storeu_pd(
                dst.as_mut_ptr().add(dst_base + quarter_n + k).cast::<f64>(),
                _mm256_add_pd(b0, c1),
            );
            _mm256_storeu_pd(
                dst.as_mut_ptr()
                    .add(dst_base + half_n + quarter_n + k)
                    .cast::<f64>(),
                _mm256_sub_pd(b0, c1),
            );
            k += 2;
        }
        while k < half_groups {
            let x0 = src[src_base + k];
            let x1 = src[src_base + half_groups + k];
            let x2 = src[src_base + groups + k] * w1;
            let x3 = src[src_base + groups + half_groups + k] * w1;
            let a0 = x0 + x2;
            let a1 = x1 + x3;
            let b0 = x0 - x2;
            let b1 = x1 - x3;
            let c0 = a1 * w2;
            let c1 = b1 * w3;
            dst[dst_base + k] = a0 + c0;
            dst[dst_base + half_n + k] = a0 - c0;
            dst[dst_base + quarter_n + k] = b0 + c1;
            dst[dst_base + half_n + quarter_n + k] = b0 - c1;
            k += 1;
        }
    }
}

/// AVX/FMA f64 two-stage Stockham leaf for `groups == 2`.
///
/// For `N = 4R`, the two fused radix-2 stages are
///
/// `a0 = x0 + w1*x2`, `a1 = x1 + w1*x3`
/// `b0 = x0 - w1*x2`, `b1 = x1 - w1*x3`
/// `y0 = a0 + w2*a1`, `y2 = a0 - w2*a1`
/// `y1 = b0 + w3*b1`, `y3 = b0 - w3*b1`.
///
/// This leaf evaluates that DAG for two adjacent Stockham digits `j` and
/// `j+1` per YMM register. The only lane rearrangement transposes the source
/// from `[x0_j,x1_j] [x0_{j+1},x1_{j+1}]` into `[x0_j,x0_{j+1}]` and
/// `[x1_j,x1_{j+1}]`, preserving independent FFT instances per complex lane.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_pair64_groups_two_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    radix: usize,
    first_twiddles: &[Complex64],
    second_twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_permute2f128_pd, _mm256_permute_pd,
        _mm256_storeu_pd, _mm256_sub_pd,
    };

    let n = src.len();
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, radix << 2);
    debug_assert_eq!(dst.len(), n);
    debug_assert!(radix >= 2);
    debug_assert_eq!(radix & 1, 0);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= 2 * radix);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let mut j = 0usize;
    while j < radix {
        let d0 = _mm256_loadu_pd(src_ptr.add(j * 4).cast::<f64>());
        let d1 = _mm256_loadu_pd(src_ptr.add((j + 1) * 4).cast::<f64>());
        let d2 = _mm256_loadu_pd(src_ptr.add(j * 4 + 2).cast::<f64>());
        let d3 = _mm256_loadu_pd(src_ptr.add((j + 1) * 4 + 2).cast::<f64>());

        let x0 = _mm256_permute2f128_pd(d0, d1, 0x20);
        let x1 = _mm256_permute2f128_pd(d0, d1, 0x31);
        let raw_x2 = _mm256_permute2f128_pd(d2, d3, 0x20);
        let raw_x3 = _mm256_permute2f128_pd(d2, d3, 0x31);

        let w1 = _mm256_loadu_pd(first_ptr.add(j).cast::<f64>());
        let w1r = _mm256_permute_pd::<0b0000>(w1);
        let w1i = _mm256_permute_pd::<0b1111>(w1);
        let x2 = cmul_vec64(w1r, w1i, raw_x2);
        let x3 = cmul_vec64(w1r, w1i, raw_x3);

        let a0 = _mm256_add_pd(x0, x2);
        let a1 = _mm256_add_pd(x1, x3);
        let b0 = _mm256_sub_pd(x0, x2);
        let b1 = _mm256_sub_pd(x1, x3);

        let w2 = _mm256_loadu_pd(second_ptr.add(j).cast::<f64>());
        let w2r = _mm256_permute_pd::<0b0000>(w2);
        let w2i = _mm256_permute_pd::<0b1111>(w2);
        let c0 = cmul_vec64(w2r, w2i, a1);

        let w3 = _mm256_loadu_pd(second_ptr.add(j + radix).cast::<f64>());
        let w3r = _mm256_permute_pd::<0b0000>(w3);
        let w3i = _mm256_permute_pd::<0b1111>(w3);
        let c1 = cmul_vec64(w3r, w3i, b1);

        _mm256_storeu_pd(dst_ptr.add(j).cast::<f64>(), _mm256_add_pd(a0, c0));
        _mm256_storeu_pd(dst_ptr.add(j + half_n).cast::<f64>(), _mm256_sub_pd(a0, c0));
        _mm256_storeu_pd(
            dst_ptr.add(j + quarter_n).cast::<f64>(),
            _mm256_add_pd(b0, c1),
        );
        _mm256_storeu_pd(
            dst_ptr.add(j + half_n + quarter_n).cast::<f64>(),
            _mm256_sub_pd(b0, c1),
        );

        j += 2;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stage_pair64_radix1_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    second_twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };

    let n = src.len();
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    let vector_end = quarter_n & !1usize;
    let w3 = second_twiddles[1];
    let w3r = _mm256_set1_pd(w3.re);
    let w3i = _mm256_set1_pd(w3.im);
    let mut k = 0usize;
    while k < vector_end {
        let x0 = _mm256_loadu_pd(src.as_ptr().add(k).cast::<f64>());
        let x1 = _mm256_loadu_pd(src.as_ptr().add(quarter_n + k).cast::<f64>());
        let x2 = _mm256_loadu_pd(src.as_ptr().add(half_n + k).cast::<f64>());
        let x3 = _mm256_loadu_pd(src.as_ptr().add(half_n + quarter_n + k).cast::<f64>());
        let a0 = _mm256_add_pd(x0, x2);
        let a1 = _mm256_add_pd(x1, x3);
        let b0 = _mm256_sub_pd(x0, x2);
        let b1 = _mm256_sub_pd(x1, x3);
        let c1 = cmul_vec64(w3r, w3i, b1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(k).cast::<f64>(), _mm256_add_pd(a0, a1));
        _mm256_storeu_pd(
            dst.as_mut_ptr().add(half_n + k).cast::<f64>(),
            _mm256_sub_pd(a0, a1),
        );
        _mm256_storeu_pd(
            dst.as_mut_ptr().add(quarter_n + k).cast::<f64>(),
            _mm256_add_pd(b0, c1),
        );
        _mm256_storeu_pd(
            dst.as_mut_ptr().add(half_n + quarter_n + k).cast::<f64>(),
            _mm256_sub_pd(b0, c1),
        );
        k += 2;
    }
    while k < quarter_n {
        let x0 = src[k];
        let x1 = src[quarter_n + k];
        let x2 = src[half_n + k];
        let x3 = src[half_n + quarter_n + k];
        let a0 = x0 + x2;
        let a1 = x1 + x3;
        let b0 = x0 - x2;
        let b1 = x1 - x3;
        let c1 = b1 * w3;
        dst[k] = a0 + a1;
        dst[half_n + k] = a0 - a1;
        dst[quarter_n + k] = b0 + c1;
        dst[half_n + quarter_n + k] = b0 - c1;
        k += 1;
    }
}
