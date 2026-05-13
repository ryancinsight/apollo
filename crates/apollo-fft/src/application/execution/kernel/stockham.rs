//! Natural-order radix-2 FFT execution through Stockham autosort.
//!
//! It computes power-of-two FFTs through a Stockham autosort pass over a
//! separate scratch buffer and writes the final spectrum in natural order. No
//! standalone reordering pass is executed before or after the butterflies.
//!
//! # Theorem
//!
//! Let `N = 2M`. Split the DFT sum into even and odd time indices:
//!
//! `X[k] = Σ_r x[2r] W_N^{2rk} + W_N^k Σ_r x[2r+1] W_N^{2rk}`.
//!
//! Since `W_N^2 = W_M`, the two sums are the `M`-point DFTs of the even and
//! odd subsequences. Therefore `X[k] = E[k] + W_N^k O[k]` and
//! `X[k+M] = E[k] - W_N^k O[k]`. The recursive routine evaluates exactly this
//! identity into caller-provided scratch storage, so the result is in natural
//! frequency order without a standalone index-reordering pass.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::empty_line_after_doc_comments)]

use super::radix_stage::{normalize_inplace_c32, normalize_inplace_c64};
use num_complex::{Complex32, Complex64};

#[cfg(target_arch = "x86_64")]
const STOCKHAM_F64_L1_RESIDENT_BYTES: usize = 32 * 1024;

#[cfg(target_arch = "x86_64")]
#[inline]
fn stockham_f64_stage_is_l1_resident(n: usize) -> bool {
    n <= STOCKHAM_F64_L1_RESIDENT_BYTES / (core::mem::size_of::<Complex64>() * 2)
}

/// L1 residency threshold for f32 triple-stage dispatch.
///
/// A three-stage Stockham pass reads src[0..n] and writes dst[0..n]. Both
/// buffers must fit in L1 to exploit the low-live codelet; 32 KiB gives 2048
/// `Complex32` elements across the two buffers.
#[cfg(target_arch = "x86_64")]
const STOCKHAM_F32_L1_RESIDENT_BYTES: usize = 32 * 1024;

#[cfg(target_arch = "x86_64")]
#[inline]
fn stockham_f32_stage_is_l1_resident(n: usize) -> bool {
    n <= STOCKHAM_F32_L1_RESIDENT_BYTES / (core::mem::size_of::<Complex32>() * 2)
}

#[inline]
fn stage_impl<C>(src: &[C], dst: &mut [C], radix: usize, twiddles: &[C])
where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    let n = src.len();
    let half_n = n >> 1;
    let groups = n / (radix << 1);
    for j in 0..radix {
        let w = twiddles[j];
        let src_base = j * groups * 2;
        let dst_base = j * groups;
        for k in 0..groups {
            let a = src[src_base + k];
            let b = src[src_base + groups + k] * w;
            dst[dst_base + k] = a + b;
            dst[dst_base + half_n + k] = a - b;
        }
    }
}

/// AVX/FMA Stockham f64 stage over two independent complex instances per vector.
///
/// For each fixed Stockham digit `j`, `k` is contiguous in both source halves
/// and destination halves:
///
/// `dst[jG + k]       = a[k] + w_j b[k]`
/// `dst[N/2 + jG + k] = a[k] - w_j b[k]`
///
/// where `G = N / (2 radix)`. Packing two `Complex64` values into one
/// `__m256d` therefore vectorizes across independent `k` instances and does
/// not mix FFT instances across lanes. The only lane permutation is the
/// within-complex real/imag swap required by complex multiplication.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage64_avx_fma(
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
unsafe fn stage64_groups_one_avx_fma(
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

/// AVX/FMA Stockham f32 stage over four independent complex instances per vector.
///
/// The vector computes `w * b` as:
///
/// `fmaddsub([wr]*b, [wi]*swap(b))`
///
/// giving even lanes `wr*br - wi*bi` and odd lanes `wr*bi + wi*br`, exactly
/// the complex product. The following vector add/sub stores the two Stockham
/// outputs. This is algebraically identical to the scalar stage and preserves
/// natural-order autosort because the destination indices are unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn stage32_avx_fma(
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
unsafe fn stage32_groups_one_avx_fma(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn cmul_vec64(
    w_re: std::arch::x86_64::__m256d,
    w_im: std::arch::x86_64::__m256d,
    value: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{_mm256_fmaddsub_pd, _mm256_mul_pd, _mm256_permute_pd};
    _mm256_fmaddsub_pd(
        w_re,
        value,
        _mm256_mul_pd(w_im, _mm256_permute_pd::<0b0101>(value)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_cmul_by_pair_twiddle(
    value: std::arch::x86_64::__m256d,
    twiddle: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::_mm256_permute_pd;

    let wr = _mm256_permute_pd::<0b0000>(twiddle);
    let wi = _mm256_permute_pd::<0b1111>(twiddle);
    cmul_vec64(wr, wi, value)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_rotate_quarter_turn(
    value: std::arch::x86_64::__m256d,
    sign_mask: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{_mm256_permute_pd, _mm256_xor_pd};

    _mm256_xor_pd(_mm256_permute_pd::<0b0101>(value), sign_mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_rotate_quarter_turn32(
    value: std::arch::x86_64::__m256,
    sign_mask: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_mm256_permute_ps, _mm256_xor_ps};

    _mm256_xor_ps(_mm256_permute_ps::<0b1011_0001>(value), sign_mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_butterfly2_pair(
    a: std::arch::x86_64::__m256d,
    b: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 2] {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_sub_pd};

    [_mm256_add_pd(a, b), _mm256_sub_pd(a, b)]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_butterfly4_pair(
    rows: [std::arch::x86_64::__m256d; 4],
    quarter_turn_mask: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 4] {
    let [mid0, mid2] = avx_butterfly2_pair(rows[0], rows[2]);
    let [mid1, mid3] = avx_butterfly2_pair(rows[1], rows[3]);
    let mid3 = avx_rotate_quarter_turn(mid3, quarter_turn_mask);
    let [out0, out1] = avx_butterfly2_pair(mid0, mid1);
    let [out2, out3] = avx_butterfly2_pair(mid2, mid3);

    [out0, out2, out1, out3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_butterfly8_pair(
    rows: [std::arch::x86_64::__m256d; 8],
    quarter_turn_mask: std::arch::x86_64::__m256d,
    w1_re: std::arch::x86_64::__m256d,
    w1_im: std::arch::x86_64::__m256d,
    w3_re: std::arch::x86_64::__m256d,
    w3_im: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 8] {
    let mid0 = avx_butterfly4_pair([rows[0], rows[2], rows[4], rows[6]], quarter_turn_mask);
    let mut mid1 = avx_butterfly4_pair([rows[1], rows[3], rows[5], rows[7]], quarter_turn_mask);
    mid1[1] = cmul_vec64(w1_re, w1_im, mid1[1]);
    mid1[2] = avx_rotate_quarter_turn(mid1[2], quarter_turn_mask);
    mid1[3] = cmul_vec64(w3_re, w3_im, mid1[3]);

    let [out0, out1] = avx_butterfly2_pair(mid0[0], mid1[0]);
    let [out2, out3] = avx_butterfly2_pair(mid0[1], mid1[1]);
    let [out4, out5] = avx_butterfly2_pair(mid0[2], mid1[2]);
    let [out6, out7] = avx_butterfly2_pair(mid0[3], mid1[3]);

    [out0, out2, out4, out6, out1, out3, out5, out7]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_transpose8_pairs(
    rows: [std::arch::x86_64::__m256d; 8],
) -> [std::arch::x86_64::__m256d; 8] {
    use std::arch::x86_64::_mm256_permute2f128_pd;

    [
        _mm256_permute2f128_pd(rows[0], rows[1], 0x20),
        _mm256_permute2f128_pd(rows[2], rows[3], 0x20),
        _mm256_permute2f128_pd(rows[4], rows[5], 0x20),
        _mm256_permute2f128_pd(rows[6], rows[7], 0x20),
        _mm256_permute2f128_pd(rows[0], rows[1], 0x31),
        _mm256_permute2f128_pd(rows[2], rows[3], 0x31),
        _mm256_permute2f128_pd(rows[4], rows[5], 0x31),
        _mm256_permute2f128_pd(rows[6], rows[7], 0x31),
    ]
}

#[inline]
unsafe fn twiddle_len64_const_unchecked<const EXPONENT: usize>(
    twiddle_ptr: *const Complex64,
) -> Complex64 {
    debug_assert!(EXPONENT < 64);
    match EXPONENT {
        0 => Complex64::new(1.0, 0.0),
        1..=31 => *twiddle_ptr.add(31 + EXPONENT),
        32 => Complex64::new(-1.0, 0.0),
        _ => {
            let w = *twiddle_ptr.add(31 + EXPONENT - 32);
            Complex64::new(-w.re, -w.im)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_twiddle_len64_pair_const<
    const FIRST_EXPONENT: usize,
    const SECOND_EXPONENT: usize,
>(
    twiddle_ptr: *const Complex64,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::_mm256_set_pd;

    let first = twiddle_len64_const_unchecked::<FIRST_EXPONENT>(twiddle_ptr);
    let second = twiddle_len64_const_unchecked::<SECOND_EXPONENT>(twiddle_ptr);
    _mm256_set_pd(second.im, second.re, first.im, first.re)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx_apply_len64_twiddles_for_column_pair<const COLUMN_PAIR: usize>(
    mid: &mut [std::arch::x86_64::__m256d; 8],
    twiddle_ptr: *const Complex64,
) {
    match COLUMN_PAIR {
        0 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len64_pair_const::<0, 1>(twiddle_ptr));
            mid[2] =
                avx_cmul_by_pair_twiddle(mid[2], avx_twiddle_len64_pair_const::<0, 2>(twiddle_ptr));
            mid[3] =
                avx_cmul_by_pair_twiddle(mid[3], avx_twiddle_len64_pair_const::<0, 3>(twiddle_ptr));
            mid[4] =
                avx_cmul_by_pair_twiddle(mid[4], avx_twiddle_len64_pair_const::<0, 4>(twiddle_ptr));
            mid[5] =
                avx_cmul_by_pair_twiddle(mid[5], avx_twiddle_len64_pair_const::<0, 5>(twiddle_ptr));
            mid[6] =
                avx_cmul_by_pair_twiddle(mid[6], avx_twiddle_len64_pair_const::<0, 6>(twiddle_ptr));
            mid[7] =
                avx_cmul_by_pair_twiddle(mid[7], avx_twiddle_len64_pair_const::<0, 7>(twiddle_ptr));
        }
        1 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len64_pair_const::<2, 3>(twiddle_ptr));
            mid[2] =
                avx_cmul_by_pair_twiddle(mid[2], avx_twiddle_len64_pair_const::<4, 6>(twiddle_ptr));
            mid[3] =
                avx_cmul_by_pair_twiddle(mid[3], avx_twiddle_len64_pair_const::<6, 9>(twiddle_ptr));
            mid[4] = avx_cmul_by_pair_twiddle(
                mid[4],
                avx_twiddle_len64_pair_const::<8, 12>(twiddle_ptr),
            );
            mid[5] = avx_cmul_by_pair_twiddle(
                mid[5],
                avx_twiddle_len64_pair_const::<10, 15>(twiddle_ptr),
            );
            mid[6] = avx_cmul_by_pair_twiddle(
                mid[6],
                avx_twiddle_len64_pair_const::<12, 18>(twiddle_ptr),
            );
            mid[7] = avx_cmul_by_pair_twiddle(
                mid[7],
                avx_twiddle_len64_pair_const::<14, 21>(twiddle_ptr),
            );
        }
        2 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len64_pair_const::<4, 5>(twiddle_ptr));
            mid[2] = avx_cmul_by_pair_twiddle(
                mid[2],
                avx_twiddle_len64_pair_const::<8, 10>(twiddle_ptr),
            );
            mid[3] = avx_cmul_by_pair_twiddle(
                mid[3],
                avx_twiddle_len64_pair_const::<12, 15>(twiddle_ptr),
            );
            mid[4] = avx_cmul_by_pair_twiddle(
                mid[4],
                avx_twiddle_len64_pair_const::<16, 20>(twiddle_ptr),
            );
            mid[5] = avx_cmul_by_pair_twiddle(
                mid[5],
                avx_twiddle_len64_pair_const::<20, 25>(twiddle_ptr),
            );
            mid[6] = avx_cmul_by_pair_twiddle(
                mid[6],
                avx_twiddle_len64_pair_const::<24, 30>(twiddle_ptr),
            );
            mid[7] = avx_cmul_by_pair_twiddle(
                mid[7],
                avx_twiddle_len64_pair_const::<28, 35>(twiddle_ptr),
            );
        }
        3 => {
            mid[1] =
                avx_cmul_by_pair_twiddle(mid[1], avx_twiddle_len64_pair_const::<6, 7>(twiddle_ptr));
            mid[2] = avx_cmul_by_pair_twiddle(
                mid[2],
                avx_twiddle_len64_pair_const::<12, 14>(twiddle_ptr),
            );
            mid[3] = avx_cmul_by_pair_twiddle(
                mid[3],
                avx_twiddle_len64_pair_const::<18, 21>(twiddle_ptr),
            );
            mid[4] = avx_cmul_by_pair_twiddle(
                mid[4],
                avx_twiddle_len64_pair_const::<24, 28>(twiddle_ptr),
            );
            mid[5] = avx_cmul_by_pair_twiddle(
                mid[5],
                avx_twiddle_len64_pair_const::<30, 35>(twiddle_ptr),
            );
            mid[6] = avx_cmul_by_pair_twiddle(
                mid[6],
                avx_twiddle_len64_pair_const::<36, 42>(twiddle_ptr),
            );
            mid[7] = avx_cmul_by_pair_twiddle(
                mid[7],
                avx_twiddle_len64_pair_const::<42, 49>(twiddle_ptr),
            );
        }
        _ => unreachable!("length-64 column pair index must be in 0..4"),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len64_first_phase_column_pair<const COLUMN_PAIR: usize>(
    data_ptr: *mut Complex64,
    scratch_ptr: *mut Complex64,
    twiddle_ptr: *const Complex64,
    quarter_turn_mask: std::arch::x86_64::__m256d,
    w1_re: std::arch::x86_64::__m256d,
    w1_im: std::arch::x86_64::__m256d,
    w3_re: std::arch::x86_64::__m256d,
    w3_im: std::arch::x86_64::__m256d,
) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};

    let column = COLUMN_PAIR << 1;
    let rows = [
        _mm256_loadu_pd(data_ptr.add(column).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 8).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 16).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 24).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 32).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 40).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 48).cast::<f64>()),
        _mm256_loadu_pd(data_ptr.add(column + 56).cast::<f64>()),
    ];
    let mut mid = avx_butterfly8_pair(rows, quarter_turn_mask, w1_re, w1_im, w3_re, w3_im);
    avx_apply_len64_twiddles_for_column_pair::<COLUMN_PAIR>(&mut mid, twiddle_ptr);
    let transposed = avx_transpose8_pairs(mid);
    let base = COLUMN_PAIR << 4;
    _mm256_storeu_pd(scratch_ptr.add(base).cast::<f64>(), transposed[0]);
    _mm256_storeu_pd(scratch_ptr.add(base + 2).cast::<f64>(), transposed[1]);
    _mm256_storeu_pd(scratch_ptr.add(base + 4).cast::<f64>(), transposed[2]);
    _mm256_storeu_pd(scratch_ptr.add(base + 6).cast::<f64>(), transposed[3]);
    _mm256_storeu_pd(scratch_ptr.add(base + 8).cast::<f64>(), transposed[4]);
    _mm256_storeu_pd(scratch_ptr.add(base + 10).cast::<f64>(), transposed[5]);
    _mm256_storeu_pd(scratch_ptr.add(base + 12).cast::<f64>(), transposed[6]);
    _mm256_storeu_pd(scratch_ptr.add(base + 14).cast::<f64>(), transposed[7]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len64_second_phase_column_pair<const COLUMN_PAIR: usize>(
    data_ptr: *mut Complex64,
    scratch_ptr: *mut Complex64,
    quarter_turn_mask: std::arch::x86_64::__m256d,
    w1_re: std::arch::x86_64::__m256d,
    w1_im: std::arch::x86_64::__m256d,
    w3_re: std::arch::x86_64::__m256d,
    w3_im: std::arch::x86_64::__m256d,
) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};

    let column = COLUMN_PAIR << 1;
    let rows = [
        _mm256_loadu_pd(scratch_ptr.add(column).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 8).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 16).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 24).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 32).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 40).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 48).cast::<f64>()),
        _mm256_loadu_pd(scratch_ptr.add(column + 56).cast::<f64>()),
    ];
    let mid = avx_butterfly8_pair(rows, quarter_turn_mask, w1_re, w1_im, w3_re, w3_im);
    _mm256_storeu_pd(data_ptr.add(column).cast::<f64>(), mid[0]);
    _mm256_storeu_pd(data_ptr.add(column + 8).cast::<f64>(), mid[1]);
    _mm256_storeu_pd(data_ptr.add(column + 16).cast::<f64>(), mid[2]);
    _mm256_storeu_pd(data_ptr.add(column + 24).cast::<f64>(), mid[3]);
    _mm256_storeu_pd(data_ptr.add(column + 32).cast::<f64>(), mid[4]);
    _mm256_storeu_pd(data_ptr.add(column + 40).cast::<f64>(), mid[5]);
    _mm256_storeu_pd(data_ptr.add(column + 48).cast::<f64>(), mid[6]);
    _mm256_storeu_pd(data_ptr.add(column + 56).cast::<f64>(), mid[7]);
}

/// Fixed-size length-64 f64 AVX/FMA base butterfly.
///
/// The kernel evaluates a 64-point DFT by the Cooley-Tukey identity with
/// `N = 8 * 8`.  Let the input be `x[8r + c]`, where `0 <= r,c < 8`.
/// Phase one computes
///
/// `A[p,c] = Σ_r x[8r+c] W_8^{pr}`,
///
/// then multiplies by `W_64^{pc}` and stores the packed transpose in `scratch`.
/// Phase two computes
///
/// `X[8q+p] = Σ_c A[p,c] W_64^{pc} W_8^{qc}`,
///
/// because `W_8^{qc} = W_64^{8qc}`.  Therefore the total exponent is
/// `c(p + 8q) + 8rp`, exactly the mixed-radix decomposition of the DFT sum.
///
/// The vector lanes hold two adjacent matrix columns, so every butterfly is
/// two independent DFT-8 instances.  The only lane movement is the required
/// 8x2 packed transpose between the two mathematical phases; no standalone
/// global reordering pass is introduced.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len64_avx_fma(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    use std::arch::x86_64::{_mm256_set1_pd, _mm256_set_pd};

    debug_assert_eq!(data.len(), 64);
    debug_assert!(scratch.len() >= 64);
    debug_assert!(twiddles.len() >= 63);

    let twiddle_ptr = twiddles.as_ptr();
    let inverse = (*twiddle_ptr.add(2)).im > 0.0;
    let quarter_turn_mask = if inverse {
        _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
    } else {
        _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
    };
    let w1 = *twiddle_ptr.add(4);
    let w3 = *twiddle_ptr.add(6);
    let w1_re = _mm256_set1_pd(w1.re);
    let w1_im = _mm256_set1_pd(w1.im);
    let w3_re = _mm256_set1_pd(w3.re);
    let w3_im = _mm256_set1_pd(w3.im);
    let data_ptr = data.as_mut_ptr();
    let scratch_ptr = scratch.as_mut_ptr();

    fixed_len64_first_phase_column_pair::<0>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_first_phase_column_pair::<1>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_first_phase_column_pair::<2>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_first_phase_column_pair::<3>(
        data_ptr,
        scratch_ptr,
        twiddle_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );

    fixed_len64_second_phase_column_pair::<0>(
        data_ptr,
        scratch_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_second_phase_column_pair::<1>(
        data_ptr,
        scratch_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_second_phase_column_pair::<2>(
        data_ptr,
        scratch_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
    fixed_len64_second_phase_column_pair::<3>(
        data_ptr,
        scratch_ptr,
        quarter_turn_mask,
        w1_re,
        w1_im,
        w3_re,
        w3_im,
    );
}

/// Fixed-size length-64 f32 AVX/FMA Stockham base wrapper.
///
/// For `N = 64 = 8 * 8`, the greedy f32 Stockham schedule consists of exactly
/// two radix-8 autosort codelets: the first consumes strides `1,2,4` and writes
/// `data -> scratch`; the second consumes strides `8,16,32` and writes
/// `scratch -> data`. The twiddle table is the same contiguous stage table used
/// by the generic scheduler:
///
/// - stage group 0 consumes `1 + 2 + 4 = 7` twiddles;
/// - stage group 1 consumes `8 + 16 + 32 = 56` twiddles.
///
/// Directly slicing those two groups is therefore algebraically identical to
/// the generic `transform::<F32StockhamAvxFma>` loop for `N = 64`, but removes
/// the schedule simulation and branch chain from the hot base case.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len64_32_avx_fma(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    debug_assert_eq!(data.len(), 64);
    debug_assert!(scratch.len() >= 64);
    debug_assert!(twiddles.len() >= 63);

    let first_second = twiddles.get_unchecked(1..3);
    let first_third = twiddles.get_unchecked(3..7);
    let second_first = twiddles.get_unchecked(7..15);
    let second_second = twiddles.get_unchecked(15..31);
    let second_third = twiddles.get_unchecked(31..63);

    stage_triple32_radix1_avx_fma(data, scratch, first_second, first_third);
    stage_triple32_quarter_groups_one_avx_fma(
        scratch,
        data,
        8,
        second_first,
        second_second,
        second_third,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn cmul_vec32(
    w_re: std::arch::x86_64::__m256,
    w_im: std::arch::x86_64::__m256,
    value: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_mm256_fmaddsub_ps, _mm256_mul_ps, _mm256_permute_ps};
    _mm256_fmaddsub_ps(
        w_re,
        value,
        _mm256_mul_ps(w_im, _mm256_permute_ps::<0b1011_0001>(value)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn cmul_pair32(
    w_re: std::arch::x86_64::__m128,
    w_im: std::arch::x86_64::__m128,
    value: std::arch::x86_64::__m128,
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::{_mm_fmaddsub_ps, _mm_mul_ps, _mm_permute_ps};
    _mm_fmaddsub_ps(
        w_re,
        value,
        _mm_mul_ps(w_im, _mm_permute_ps::<0b1011_0001>(value)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn store_complex32_low(dst: *mut Complex32, value: std::arch::x86_64::__m128) {
    use std::arch::x86_64::{_mm_castps_si128, _mm_storel_epi64};
    _mm_storel_epi64(dst.cast(), _mm_castps_si128(value));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn store_complex32_high(dst: *mut Complex32, value: std::arch::x86_64::__m128) {
    use std::arch::x86_64::_mm_movehl_ps;
    store_complex32_low(dst, _mm_movehl_ps(value, value));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_pair64_avx_fma(
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
unsafe fn stage_pair64_groups_two_avx_fma(
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
#[inline]
unsafe fn stage_pair32_avx_fma(
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
unsafe fn stage_pair64_radix1_avx_fma(
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

/// Fuses the first three f64 Stockham stages when the incoming radix is one.
///
/// For `radix = 1`, the three-stage Stockham substitution has fixed roots:
/// `W2^0 = 1`, `W4^0 = 1`, `W8^0 = 1`, and `W8^2 = W4^1`. Direct substitution
/// into the scalar recurrence therefore removes the stage-1 twiddle product,
/// the stage-2 zero-exponent product, and the stage-3 zero-exponent product.
/// The remaining `W4^1` and `W8^2` products are quarter-turn rotations, while
/// `W8^1` and `W8^3` stay as FMA complex products. Output indices are identical
/// to `stage_triple64` with `radix = 1`, so Stockham autosort order is preserved.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_triple64_radix1_avx_fma(
    src: &[Complex64],
    dst: &mut [Complex64],
    second_twiddles: &[Complex64],
    third_twiddles: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_set_pd, _mm256_storeu_pd,
        _mm256_sub_pd,
    };

    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert!(n >= 16);
    debug_assert_eq!(n & (n - 1), 0);
    debug_assert!(second_twiddles.len() >= 2);
    debug_assert!(third_twiddles.len() >= 4);
    debug_assert_eq!(eighth_n & 1, 0);

    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();
    let w2b = *second_ptr.add(1);
    let w3b = *third_ptr.add(1);
    let w3c = *third_ptr.add(2);
    let w3d = *third_ptr.add(3);
    let w3br = _mm256_set1_pd(w3b.re);
    let w3bi = _mm256_set1_pd(w3b.im);
    let w3dr = _mm256_set1_pd(w3d.re);
    let w3di = _mm256_set1_pd(w3d.im);
    let w2_quarter_turn_mask = if w2b.im > 0.0 {
        _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
    } else {
        _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
    };
    let w3_quarter_turn_mask = if w3c.im > 0.0 {
        _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
    } else {
        _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
    };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mut k = 0usize;
    while k < eighth_n {
        let x0 = _mm256_loadu_pd(src_ptr.add(k).cast::<f64>());
        let x2 = _mm256_loadu_pd(src_ptr.add(2 * eighth_n + k).cast::<f64>());
        let x4 = _mm256_loadu_pd(src_ptr.add(4 * eighth_n + k).cast::<f64>());
        let x6 = _mm256_loadu_pd(src_ptr.add(6 * eighth_n + k).cast::<f64>());

        let s0 = _mm256_add_pd(x0, x4);
        let s2 = _mm256_add_pd(x2, x6);
        let d0 = _mm256_sub_pd(x0, x4);
        let d2 = _mm256_sub_pd(x2, x6);

        let u2 = avx_rotate_quarter_turn(d2, w2_quarter_turn_mask);
        let p0 = _mm256_add_pd(s0, s2);
        let p2 = _mm256_add_pd(d0, u2);
        let p4 = _mm256_sub_pd(s0, s2);
        let p6 = _mm256_sub_pd(d0, u2);

        let x1 = _mm256_loadu_pd(src_ptr.add(eighth_n + k).cast::<f64>());
        let x3 = _mm256_loadu_pd(src_ptr.add(3 * eighth_n + k).cast::<f64>());
        let x5 = _mm256_loadu_pd(src_ptr.add(5 * eighth_n + k).cast::<f64>());
        let x7 = _mm256_loadu_pd(src_ptr.add(7 * eighth_n + k).cast::<f64>());

        let s1 = _mm256_add_pd(x1, x5);
        let s3 = _mm256_add_pd(x3, x7);
        let d1 = _mm256_sub_pd(x1, x5);
        let d3 = _mm256_sub_pd(x3, x7);
        let u3 = avx_rotate_quarter_turn(d3, w2_quarter_turn_mask);
        let p1 = _mm256_add_pd(s1, s3);
        let p3 = _mm256_add_pd(d1, u3);
        let p5 = _mm256_sub_pd(s1, s3);
        let p7 = _mm256_sub_pd(d1, u3);

        let q2 = avx_rotate_quarter_turn(p5, w3_quarter_turn_mask);
        _mm256_storeu_pd(dst_ptr.add(k).cast::<f64>(), _mm256_add_pd(p0, p1));
        _mm256_storeu_pd(dst_ptr.add(half_n + k).cast::<f64>(), _mm256_sub_pd(p0, p1));
        _mm256_storeu_pd(
            dst_ptr.add(quarter_n + k).cast::<f64>(),
            _mm256_add_pd(p4, q2),
        );
        _mm256_storeu_pd(
            dst_ptr.add(half_n + quarter_n + k).cast::<f64>(),
            _mm256_sub_pd(p4, q2),
        );

        let q1 = cmul_vec64(w3br, w3bi, p3);
        let q3 = cmul_vec64(w3dr, w3di, p7);
        _mm256_storeu_pd(
            dst_ptr.add(eighth_n + k).cast::<f64>(),
            _mm256_add_pd(p2, q1),
        );
        _mm256_storeu_pd(
            dst_ptr.add(half_n + eighth_n + k).cast::<f64>(),
            _mm256_sub_pd(p2, q1),
        );
        _mm256_storeu_pd(
            dst_ptr.add(quarter_n + eighth_n + k).cast::<f64>(),
            _mm256_add_pd(p6, q3),
        );
        _mm256_storeu_pd(
            dst_ptr.add(half_n + quarter_n + eighth_n + k).cast::<f64>(),
            _mm256_sub_pd(p6, q3),
        );
        k += 2;
    }
}

/// Fuses the first three f32 Stockham stages when the incoming radix is one.
///
/// For `radix = 1`, the first-stage twiddle and the lower second/third-stage
/// twiddles are exactly one. The upper second-stage twiddle is a quarter-turn,
/// and the third-stage `W8^2` branch is also a quarter-turn. Direct substitution
/// removes those complex products and keeps only the `W8^1` and `W8^3` FMA
/// products. The function is the f32 native-precision analogue of
/// `stage_triple64_radix1_avx_fma`; it vectorizes across independent `k`
/// instances and writes the same eight Stockham autosort bands.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_triple32_radix1_avx_fma(
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
#[inline]
unsafe fn stage_pair32_radix1_avx_fma(
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
unsafe fn stage_triple64_quarter_groups_one_avx_fma(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_triple64_low_live_avx_fma(
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
            let x2 = _mm256_loadu_pd(src_ptr.add(src_base + 2 * quarter_groups + k).cast::<f64>());
            let x4 = cmul_vec64(
                w1r,
                w1i,
                _mm256_loadu_pd(src_ptr.add(src_base + groups + k).cast::<f64>()),
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

            let s0 = _mm256_add_pd(x0, x4);
            let s2 = _mm256_add_pd(x2, x6);
            let d0 = _mm256_sub_pd(x0, x4);
            let d2 = _mm256_sub_pd(x2, x6);

            let t2 = cmul_vec64(w2ar, w2ai, s2);
            let p0 = _mm256_add_pd(s0, t2);
            let p4 = _mm256_sub_pd(s0, t2);
            let x1 = _mm256_loadu_pd(src_ptr.add(src_base + quarter_groups + k).cast::<f64>());
            let x3 = _mm256_loadu_pd(src_ptr.add(src_base + 3 * quarter_groups + k).cast::<f64>());
            let x5 = cmul_vec64(
                w1r,
                w1i,
                _mm256_loadu_pd(
                    src_ptr
                        .add(src_base + groups + quarter_groups + k)
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
            let s1 = _mm256_add_pd(x1, x5);
            let s3 = _mm256_add_pd(x3, x7);
            let d1 = _mm256_sub_pd(x1, x5);
            let d3 = _mm256_sub_pd(x3, x7);
            let t3 = cmul_vec64(w2ar, w2ai, s3);
            let p1 = _mm256_add_pd(s1, t3);
            let p5 = _mm256_sub_pd(s1, t3);
            let out_base = dst_base + k;
            let q0 = cmul_vec64(w3ar, w3ai, p1);
            let q2 = cmul_vec64(w3cr, w3ci, p5);
            _mm256_storeu_pd(dst_ptr.add(out_base).cast::<f64>(), _mm256_add_pd(p0, q0));
            _mm256_storeu_pd(
                dst_ptr.add(half_n + out_base).cast::<f64>(),
                _mm256_sub_pd(p0, q0),
            );
            _mm256_storeu_pd(
                dst_ptr.add(quarter_n + out_base).cast::<f64>(),
                _mm256_add_pd(p4, q2),
            );
            _mm256_storeu_pd(
                dst_ptr.add(half_n + quarter_n + out_base).cast::<f64>(),
                _mm256_sub_pd(p4, q2),
            );

            let u2 = cmul_vec64(w2br, w2bi, d2);
            let u3 = cmul_vec64(w2br, w2bi, d3);
            let p2 = _mm256_add_pd(d0, u2);
            let p3 = _mm256_add_pd(d1, u3);
            let p6 = _mm256_sub_pd(d0, u2);
            let p7 = _mm256_sub_pd(d1, u3);
            let q1 = cmul_vec64(w3br, w3bi, p3);
            let q3 = cmul_vec64(w3dr, w3di, p7);
            _mm256_storeu_pd(
                dst_ptr.add(eighth_n + out_base).cast::<f64>(),
                _mm256_add_pd(p2, q1),
            );
            _mm256_storeu_pd(
                dst_ptr.add(half_n + eighth_n + out_base).cast::<f64>(),
                _mm256_sub_pd(p2, q1),
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

/// Fuses a f64 three-stage Stockham suffix when `groups == 8`.
///
/// With `G = 8`, each Stockham digit `j` owns sixteen contiguous complex
/// source values. The scalar three-stage recurrence is
///
/// `s_i = x_i + w1*x_{i+4}`, `d_i = x_i - w1*x_{i+4}`,
/// `p_{0,1,4,5} = s_{0,1} ± w2a*s_{2,3}`,
/// `p_{2,3,6,7} = d_{0,1} ± w2b*d_{2,3}`,
/// followed by `p_even ± w3*p_odd` into the eight Stockham output bands.
///
/// This leaf is the `G = 8` specialization of `stage_triple64_throughput_avx_fma`.
/// The loop trip count over `k` is statically one YMM vector, and late twiddle
/// broadcasts are emitted at first use. The mathematical DAG is unchanged, so
/// every output index and twiddle exponent is identical to the generic fused
/// radix-8 Stockham stage without adding a permutation pass.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_triple64_groups_eight_avx_fma(
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
unsafe fn stage_triple64_throughput_avx_fma(
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

mod private {
    pub(super) trait Sealed {}
}

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
struct F64Stockham;
#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
struct F32Stockham;
struct StockhamFused1;
struct StockhamFused2;
struct StockhamFused3;
struct StockhamFused4;

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl private::Sealed for F64Stockham {}
#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl private::Sealed for F32Stockham {}
impl private::Sealed for StockhamFused1 {}
impl private::Sealed for StockhamFused2 {}
impl private::Sealed for StockhamFused3 {}
impl private::Sealed for StockhamFused4 {}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_pair32_quarter_groups_two_avx_fma(
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
unsafe fn stage_pair32_groups_two_avx_fma(
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

/// AVX/FMA f32 three-stage fused Stockham radix-8 kernel.
///
/// Mirrors `stage_triple64_throughput_avx_fma` for Complex32. Each YMM register
/// holds four Complex32 values (8 f32 = 256 bits), so the vectorized inner loop
/// advances `k` by 4. Dispatch routes only `quarter_groups % 4 == 0` shapes to
/// this leaf, which keeps the hot codelet free of scalar-tail control flow.
///
/// ## Correctness
///
/// The butterfly DAG is identical to `stage_triple64_throughput_avx_fma` under
/// complex-field substitution f64 → f32. The three fused stages compute:
///
/// Stage 1: `x_{4..7} *= w1[j]`; form `s_{0..3} = x_{0..3} + x_{4..7}`,
///          `d_{0..3} = x_{0..3} - x_{4..7}`.
/// Stage 2: `t_{2,3} = s_{2,3} * w2a[j]`, `u_{2,3} = d_{2,3} * w2b[j]`.
/// Stage 3: four independent twiddle products `q_{0..3}` with `w3{a..d}[j]`.
///
/// Outputs are written to the eight stride-1/8 interleaved destination bands,
/// identical to the scalar `stage_triple32` (and by extension `stage_triple64`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_triple32_avx_fma(
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
unsafe fn stage_triple32_low_live_avx_fma(
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

/// Fuses a f32 three-stage Stockham suffix when `groups == 8`.
///
/// With `G = 8`, `quarter_groups = 2`, so each digit `j` owns 16 contiguous
/// source values. The 8 vectors (`x0..x7`) are mapped exactly to four
/// 256-bit registers: `[x0, x1]`, `[x2, x3]`, `[x4, x5]`, `[x6, x7]`.
/// This leaf evaluates the same DAG in vector registers without loop control.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_triple32_quarter_groups_two_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
    third_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castpd_ps, _mm256_castps256_ps128, _mm256_castps_pd,
        _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_permute2f128_pd, _mm256_set1_ps,
        _mm256_set_ps, _mm256_sub_ps, _mm_storeu_ps,
    };

    let n = src.len();
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;
    debug_assert_eq!(n, radix << 4);
    debug_assert_eq!(dst.len(), n);
    debug_assert!(radix >= 1);

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

        let src_base = j * 16;
        let y01 = _mm256_loadu_ps(src_ptr.add(src_base).cast::<f32>());
        let y23 = _mm256_loadu_ps(src_ptr.add(src_base + 4).cast::<f32>());
        let y45 = _mm256_loadu_ps(src_ptr.add(src_base + 8).cast::<f32>());
        let y67 = _mm256_loadu_ps(src_ptr.add(src_base + 12).cast::<f32>());

        let x45 = cmul_vec32(w1r, w1i, y45);
        let x67 = cmul_vec32(w1r, w1i, y67);

        let s01 = _mm256_add_ps(y01, x45);
        let s23 = _mm256_add_ps(y23, x67);
        let d01 = _mm256_sub_ps(y01, x45);
        let d23 = _mm256_sub_ps(y23, x67);

        let t23 = cmul_vec32(w2ar, w2ai, s23);
        let u23 = cmul_vec32(w2br, w2bi, d23);

        let p01 = _mm256_add_ps(s01, t23);
        let p45 = _mm256_sub_ps(s01, t23);
        let p23 = _mm256_add_ps(d01, u23);
        let p67 = _mm256_sub_ps(d01, u23);

        let p02 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(p01),
            _mm256_castps_pd(p23),
            0x20,
        ));
        let p13 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(p01),
            _mm256_castps_pd(p23),
            0x31,
        ));

        let w3ab_r = _mm256_set_ps(
            w3b.re, w3b.re, w3b.re, w3b.re, w3a.re, w3a.re, w3a.re, w3a.re,
        );
        let w3ab_i = _mm256_set_ps(
            w3b.im, w3b.im, w3b.im, w3b.im, w3a.im, w3a.im, w3a.im, w3a.im,
        );
        let q01 = cmul_vec32(w3ab_r, w3ab_i, p13);

        let p46 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(p45),
            _mm256_castps_pd(p67),
            0x20,
        ));
        let p57 = _mm256_castpd_ps(_mm256_permute2f128_pd(
            _mm256_castps_pd(p45),
            _mm256_castps_pd(p67),
            0x31,
        ));

        let w3cd_r = _mm256_set_ps(
            w3d.re, w3d.re, w3d.re, w3d.re, w3c.re, w3c.re, w3c.re, w3c.re,
        );
        let w3cd_i = _mm256_set_ps(
            w3d.im, w3d.im, w3d.im, w3d.im, w3c.im, w3c.im, w3c.im, w3c.im,
        );
        let q23 = cmul_vec32(w3cd_r, w3cd_i, p57);

        let out02 = _mm256_add_ps(p02, q01);
        let out13 = _mm256_sub_ps(p02, q01);
        let out46 = _mm256_add_ps(p46, q23);
        let out57 = _mm256_sub_ps(p46, q23);

        let out_base = j * 2;
        _mm_storeu_ps(
            dst_ptr.add(out_base).cast::<f32>(),
            _mm256_castps256_ps128(out02),
        );
        _mm_storeu_ps(
            dst_ptr.add(eighth_n + out_base).cast::<f32>(),
            _mm256_extractf128_ps(out02, 1),
        );
        _mm_storeu_ps(
            dst_ptr.add(half_n + out_base).cast::<f32>(),
            _mm256_castps256_ps128(out13),
        );
        _mm_storeu_ps(
            dst_ptr.add(half_n + eighth_n + out_base).cast::<f32>(),
            _mm256_extractf128_ps(out13, 1),
        );

        _mm_storeu_ps(
            dst_ptr.add(quarter_n + out_base).cast::<f32>(),
            _mm256_castps256_ps128(out46),
        );
        _mm_storeu_ps(
            dst_ptr.add(quarter_n + eighth_n + out_base).cast::<f32>(),
            _mm256_extractf128_ps(out46, 1),
        );
        _mm_storeu_ps(
            dst_ptr.add(half_n + quarter_n + out_base).cast::<f32>(),
            _mm256_castps256_ps128(out57),
        );
        _mm_storeu_ps(
            dst_ptr
                .add(half_n + quarter_n + eighth_n + out_base)
                .cast::<f32>(),
            _mm256_extractf128_ps(out57, 1),
        );
    }
}

/// Fuses a f32 three-stage Stockham suffix when `groups == 4`.
///
/// With `G = 4`, each digit `j` owns exactly eight contiguous source values
/// `x0..x7`, and `quarter_groups = 1`. The scalar `stage_triple32` recurrence
/// therefore stores one complex result into each eighth-band at offset `j`.
/// This leaf evaluates the same DAG using 256-bit AVX registers by processing
/// two consecutive `j` values per iteration:
///
/// Lower 128 bits carry the data and twiddles for `j`, upper 128 bits for `j+1`.
/// Within each 128-bit lane the butterfly DAG is identical:
///
/// `s = x0..x3 + w1*x4..x7`, `d = x0..x3 - w1*x4..x7`;
/// `p0,p1,p4,p5` are produced from the `s` half and `w2a`;
/// `p2,p3,p6,p7` are produced from the `d` half and `w2b`;
/// the stage-3 products consume only `p1,p3,p5,p7`.
///
/// `_mm256_shuffle_ps` applies the same within-lane permutation to both halves
/// simultaneously, so `[p0,p2]`/`[p1,p3]` are extracted for both `j` values
/// with a single instruction. An SSE scalar tail handles odd `radix`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stage_triple32_quarter_groups_one_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
    third_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castps128_ps256, _mm256_castps256_ps128, _mm256_extractf128_ps,
        _mm256_insertf128_ps, _mm256_set_ps, _mm256_shuffle_ps, _mm256_sub_ps, _mm_add_ps,
        _mm_loadu_ps, _mm_set1_ps, _mm_set_ps, _mm_shuffle_ps, _mm_sub_ps,
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

    // 256-bit path: pack j and j+1 into lower/upper 128-bit halves of each YMM.
    // Each j owns 8 Complex32 = 16 f32 = 1 cache line; two adjacent j's are loaded
    // with 8 XMM loads assembled into 4 YMM registers via insertf128.
    let vector_end = radix & !1;
    let mut j = 0usize;
    while j < vector_end {
        let w1_j0 = *first_ptr.add(j);
        let w1_j1 = *first_ptr.add(j + 1);
        let w2a_j0 = *second_ptr.add(j);
        let w2a_j1 = *second_ptr.add(j + 1);
        let w2b_j0 = *second_ptr.add(j + radix);
        let w2b_j1 = *second_ptr.add(j + 1 + radix);
        let w3a_j0 = *third_ptr.add(j);
        let w3a_j1 = *third_ptr.add(j + 1);
        let w3b_j0 = *third_ptr.add(j + radix);
        let w3b_j1 = *third_ptr.add(j + 1 + radix);
        let w3c_j0 = *third_ptr.add(j + 2 * radix);
        let w3c_j1 = *third_ptr.add(j + 1 + 2 * radix);
        let w3d_j0 = *third_ptr.add(j + 3 * radix);
        let w3d_j1 = *third_ptr.add(j + 1 + 3 * radix);

        // Load: src[j*8..j*8+8] into lower 128, src[(j+1)*8..(j+1)*8+8] into upper 128.
        let src_base = j * 8;
        let lo01 = _mm_loadu_ps(src_ptr.add(src_base).cast::<f32>());
        let lo23 = _mm_loadu_ps(src_ptr.add(src_base + 2).cast::<f32>());
        let lo45 = _mm_loadu_ps(src_ptr.add(src_base + 4).cast::<f32>());
        let lo67 = _mm_loadu_ps(src_ptr.add(src_base + 6).cast::<f32>());
        let hi01 = _mm_loadu_ps(src_ptr.add(src_base + 8).cast::<f32>());
        let hi23 = _mm_loadu_ps(src_ptr.add(src_base + 10).cast::<f32>());
        let hi45 = _mm_loadu_ps(src_ptr.add(src_base + 12).cast::<f32>());
        let hi67 = _mm_loadu_ps(src_ptr.add(src_base + 14).cast::<f32>());
        let ymm01 = _mm256_insertf128_ps(_mm256_castps128_ps256(lo01), hi01, 1);
        let ymm23 = _mm256_insertf128_ps(_mm256_castps128_ps256(lo23), hi23, 1);
        let ymm45 = _mm256_insertf128_ps(_mm256_castps128_ps256(lo45), hi45, 1);
        let ymm67 = _mm256_insertf128_ps(_mm256_castps128_ps256(lo67), hi67, 1);

        // Twiddle broadcast: elements [0..3] = j0 broadcast, elements [4..7] = j1 broadcast.
        // Each Complex32 occupies 2 f32 slots, so each twiddle scalar is replicated twice per lane.
        let ymm_w1r = _mm256_set_ps(
            w1_j1.re, w1_j1.re, w1_j1.re, w1_j1.re, w1_j0.re, w1_j0.re, w1_j0.re, w1_j0.re,
        );
        let ymm_w1i = _mm256_set_ps(
            w1_j1.im, w1_j1.im, w1_j1.im, w1_j1.im, w1_j0.im, w1_j0.im, w1_j0.im, w1_j0.im,
        );
        let ymm_w2ar = _mm256_set_ps(
            w2a_j1.re, w2a_j1.re, w2a_j1.re, w2a_j1.re, w2a_j0.re, w2a_j0.re, w2a_j0.re, w2a_j0.re,
        );
        let ymm_w2ai = _mm256_set_ps(
            w2a_j1.im, w2a_j1.im, w2a_j1.im, w2a_j1.im, w2a_j0.im, w2a_j0.im, w2a_j0.im, w2a_j0.im,
        );
        let ymm_w2br = _mm256_set_ps(
            w2b_j1.re, w2b_j1.re, w2b_j1.re, w2b_j1.re, w2b_j0.re, w2b_j0.re, w2b_j0.re, w2b_j0.re,
        );
        let ymm_w2bi = _mm256_set_ps(
            w2b_j1.im, w2b_j1.im, w2b_j1.im, w2b_j1.im, w2b_j0.im, w2b_j0.im, w2b_j0.im, w2b_j0.im,
        );

        let x45 = cmul_vec32(ymm_w1r, ymm_w1i, ymm45);
        let x67 = cmul_vec32(ymm_w1r, ymm_w1i, ymm67);

        let s01 = _mm256_add_ps(ymm01, x45);
        let s23 = _mm256_add_ps(ymm23, x67);
        let d01 = _mm256_sub_ps(ymm01, x45);
        let d23 = _mm256_sub_ps(ymm23, x67);

        let t23 = cmul_vec32(ymm_w2ar, ymm_w2ai, s23);
        let u23 = cmul_vec32(ymm_w2br, ymm_w2bi, d23);
        let p01 = _mm256_add_ps(s01, t23);
        let p45 = _mm256_sub_ps(s01, t23);
        let p23 = _mm256_add_ps(d01, u23);
        let p67 = _mm256_sub_ps(d01, u23);

        // Within-lane shuffle: same imm8 applied independently to j0 (low) and j1 (high).
        // 0b1110_1110 = select [a[2],a[3],b[2],b[3]] per lane → extracts p1 and p3.
        // 0b0100_0100 = select [a[0],a[1],b[0],b[1]] per lane → extracts p0 and p2.
        let p13 = _mm256_shuffle_ps::<0b1110_1110>(p01, p23);
        let p02 = _mm256_shuffle_ps::<0b0100_0100>(p01, p23);

        // w3ab broadcast: elements [0,1] = w3a for j0's p1, [2,3] = w3b for j0's p3,
        //                 [4,5] = w3a for j1's p1, [6,7] = w3b for j1's p3.
        let ymm_w3ab_r = _mm256_set_ps(
            w3b_j1.re, w3b_j1.re, w3a_j1.re, w3a_j1.re, w3b_j0.re, w3b_j0.re, w3a_j0.re, w3a_j0.re,
        );
        let ymm_w3ab_i = _mm256_set_ps(
            w3b_j1.im, w3b_j1.im, w3a_j1.im, w3a_j1.im, w3b_j0.im, w3b_j0.im, w3a_j0.im, w3a_j0.im,
        );
        let q01 = cmul_vec32(ymm_w3ab_r, ymm_w3ab_i, p13);
        let out02 = _mm256_add_ps(p02, q01);
        let out13 = _mm256_sub_ps(p02, q01);

        let p57 = _mm256_shuffle_ps::<0b1110_1110>(p45, p67);
        let p46 = _mm256_shuffle_ps::<0b0100_0100>(p45, p67);
        let ymm_w3cd_r = _mm256_set_ps(
            w3d_j1.re, w3d_j1.re, w3c_j1.re, w3c_j1.re, w3d_j0.re, w3d_j0.re, w3c_j0.re, w3c_j0.re,
        );
        let ymm_w3cd_i = _mm256_set_ps(
            w3d_j1.im, w3d_j1.im, w3c_j1.im, w3c_j1.im, w3d_j0.im, w3d_j0.im, w3c_j0.im, w3c_j0.im,
        );
        let q23 = cmul_vec32(ymm_w3cd_r, ymm_w3cd_i, p57);
        let out46 = _mm256_add_ps(p46, q23);
        let out57 = _mm256_sub_ps(p46, q23);

        // Unpack lower (j0) and upper (j1) halves and store each complex to its band.
        let lo02 = _mm256_castps256_ps128(out02);
        let hi02 = _mm256_extractf128_ps(out02, 1);
        store_complex32_low(dst_ptr.add(j), lo02);
        store_complex32_low(dst_ptr.add(j + 1), hi02);
        store_complex32_high(dst_ptr.add(eighth_n + j), lo02);
        store_complex32_high(dst_ptr.add(eighth_n + j + 1), hi02);

        let lo13 = _mm256_castps256_ps128(out13);
        let hi13 = _mm256_extractf128_ps(out13, 1);
        store_complex32_low(dst_ptr.add(half_n + j), lo13);
        store_complex32_low(dst_ptr.add(half_n + j + 1), hi13);
        store_complex32_high(dst_ptr.add(half_n + eighth_n + j), lo13);
        store_complex32_high(dst_ptr.add(half_n + eighth_n + j + 1), hi13);

        let lo46 = _mm256_castps256_ps128(out46);
        let hi46 = _mm256_extractf128_ps(out46, 1);
        store_complex32_low(dst_ptr.add(quarter_n + j), lo46);
        store_complex32_low(dst_ptr.add(quarter_n + j + 1), hi46);
        store_complex32_high(dst_ptr.add(quarter_n + eighth_n + j), lo46);
        store_complex32_high(dst_ptr.add(quarter_n + eighth_n + j + 1), hi46);

        let lo57 = _mm256_castps256_ps128(out57);
        let hi57 = _mm256_extractf128_ps(out57, 1);
        store_complex32_low(dst_ptr.add(half_n + quarter_n + j), lo57);
        store_complex32_low(dst_ptr.add(half_n + quarter_n + j + 1), hi57);
        store_complex32_high(dst_ptr.add(half_n + quarter_n + eighth_n + j), lo57);
        store_complex32_high(dst_ptr.add(half_n + quarter_n + eighth_n + j + 1), hi57);

        j += 2;
    }

    // SSE scalar tail for odd radix (at most one iteration).
    while j < radix {
        let w1 = *first_ptr.add(j);
        let w2a = *second_ptr.add(j);
        let w2b = *second_ptr.add(j + radix);
        let w3a = *third_ptr.add(j);
        let w3b = *third_ptr.add(j + radix);
        let w3c = *third_ptr.add(j + 2 * radix);
        let w3d = *third_ptr.add(j + 3 * radix);

        let w1r = _mm_set1_ps(w1.re);
        let w1i = _mm_set1_ps(w1.im);
        let src_base = j * 8;
        let x01 = _mm_loadu_ps(src_ptr.add(src_base).cast::<f32>());
        let x23 = _mm_loadu_ps(src_ptr.add(src_base + 2).cast::<f32>());
        let x45 = cmul_pair32(
            w1r,
            w1i,
            _mm_loadu_ps(src_ptr.add(src_base + 4).cast::<f32>()),
        );
        let x67 = cmul_pair32(
            w1r,
            w1i,
            _mm_loadu_ps(src_ptr.add(src_base + 6).cast::<f32>()),
        );

        let s01 = _mm_add_ps(x01, x45);
        let s23 = _mm_add_ps(x23, x67);
        let d01 = _mm_sub_ps(x01, x45);
        let d23 = _mm_sub_ps(x23, x67);

        let t23 = cmul_pair32(_mm_set1_ps(w2a.re), _mm_set1_ps(w2a.im), s23);
        let u23 = cmul_pair32(_mm_set1_ps(w2b.re), _mm_set1_ps(w2b.im), d23);
        let p01 = _mm_add_ps(s01, t23);
        let p45 = _mm_sub_ps(s01, t23);
        let p23 = _mm_add_ps(d01, u23);
        let p67 = _mm_sub_ps(d01, u23);

        let p13 = _mm_shuffle_ps::<0b1110_1110>(p01, p23);
        let q01 = cmul_pair32(
            _mm_set_ps(w3b.re, w3b.re, w3a.re, w3a.re),
            _mm_set_ps(w3b.im, w3b.im, w3a.im, w3a.im),
            p13,
        );
        let p02 = _mm_shuffle_ps::<0b0100_0100>(p01, p23);
        let out02 = _mm_add_ps(p02, q01);
        let out13 = _mm_sub_ps(p02, q01);
        store_complex32_low(dst_ptr.add(j), out02);
        store_complex32_low(dst_ptr.add(half_n + j), out13);
        store_complex32_high(dst_ptr.add(eighth_n + j), out02);
        store_complex32_high(dst_ptr.add(half_n + eighth_n + j), out13);

        let p57 = _mm_shuffle_ps::<0b1110_1110>(p45, p67);
        let q23 = cmul_pair32(
            _mm_set_ps(w3d.re, w3d.re, w3c.re, w3c.re),
            _mm_set_ps(w3d.im, w3d.im, w3c.im, w3c.im),
            p57,
        );
        let p46 = _mm_shuffle_ps::<0b0100_0100>(p45, p67);
        let out46 = _mm_add_ps(p46, q23);
        let out57 = _mm_sub_ps(p46, q23);
        store_complex32_low(dst_ptr.add(quarter_n + j), out46);
        store_complex32_low(dst_ptr.add(half_n + quarter_n + j), out57);
        store_complex32_high(dst_ptr.add(quarter_n + eighth_n + j), out46);
        store_complex32_high(dst_ptr.add(half_n + quarter_n + eighth_n + j), out57);

        j += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stockham_quad_store_pair64(
    dst_ptr: *mut Complex64,
    low_index: usize,
    high_index: usize,
    pair: std::arch::x86_64::__m256d,
    twiddle: Complex64,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_castpd256_pd128, _mm256_permute2f128_pd, _mm256_set1_pd,
        _mm256_sub_pd, _mm_storeu_pd,
    };

    let odd = _mm256_permute2f128_pd(pair, pair, 0x11);
    let product = cmul_vec64(_mm256_set1_pd(twiddle.re), _mm256_set1_pd(twiddle.im), odd);
    _mm_storeu_pd(
        dst_ptr.add(low_index).cast::<f64>(),
        _mm256_castpd256_pd128(_mm256_add_pd(pair, product)),
    );
    _mm_storeu_pd(
        dst_ptr.add(high_index).cast::<f64>(),
        _mm256_castpd256_pd128(_mm256_sub_pd(pair, product)),
    );
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct StockhamQuadFirstPairs64 {
    p01: std::arch::x86_64::__m256d,
    p45: std::arch::x86_64::__m256d,
    p89: std::arch::x86_64::__m256d,
    p12_13: std::arch::x86_64::__m256d,
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct StockhamQuadSecondPairs64 {
    p23: std::arch::x86_64::__m256d,
    p67: std::arch::x86_64::__m256d,
    p10_11: std::arch::x86_64::__m256d,
    p14_15: std::arch::x86_64::__m256d,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stockham_quad_groups_eight64_first_pairs(
    src_ptr: *const Complex64,
    first_ptr: *const Complex64,
    second_ptr: *const Complex64,
    third_ptr: *const Complex64,
    radix: usize,
    j: usize,
) -> StockhamQuadFirstPairs64 {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_sub_pd};

    let w1 = *first_ptr.add(j);
    let w20 = *second_ptr.add(j);
    let w1r = _mm256_set1_pd(w1.re);
    let w1i = _mm256_set1_pd(w1.im);
    let w20r = _mm256_set1_pd(w20.re);
    let w20i = _mm256_set1_pd(w20.im);

    let src_base = j << 4;
    let x01 = _mm256_loadu_pd(src_ptr.add(src_base).cast::<f64>());
    let x23 = _mm256_loadu_pd(src_ptr.add(src_base + 2).cast::<f64>());
    let x45 = _mm256_loadu_pd(src_ptr.add(src_base + 4).cast::<f64>());
    let x67 = _mm256_loadu_pd(src_ptr.add(src_base + 6).cast::<f64>());
    let x89 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 8).cast::<f64>()),
    );
    let x10_11 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 10).cast::<f64>()),
    );
    let x12_13 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 12).cast::<f64>()),
    );
    let x14_15 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 14).cast::<f64>()),
    );

    let y01 = _mm256_add_pd(x01, x89);
    let y23 = _mm256_add_pd(x23, x10_11);
    let y45 = _mm256_add_pd(x45, x12_13);
    let y67 = _mm256_add_pd(x67, x14_15);
    let t45 = cmul_vec64(w20r, w20i, y45);
    let t67 = cmul_vec64(w20r, w20i, y67);

    let z01 = _mm256_add_pd(y01, t45);
    let z23 = _mm256_add_pd(y23, t67);
    let z89 = _mm256_sub_pd(y01, t45);
    let z10_11 = _mm256_sub_pd(y23, t67);

    let w30 = *third_ptr.add(j);
    let w30r = _mm256_set1_pd(w30.re);
    let w30i = _mm256_set1_pd(w30.im);
    let u23 = cmul_vec64(w30r, w30i, z23);
    let p01 = _mm256_add_pd(z01, u23);
    let p89 = _mm256_sub_pd(z01, u23);

    let w32 = *third_ptr.add(j + 2 * radix);
    let w32r = _mm256_set1_pd(w32.re);
    let w32i = _mm256_set1_pd(w32.im);
    let u10_11 = cmul_vec64(w32r, w32i, z10_11);
    let p45 = _mm256_add_pd(z89, u10_11);
    let p12_13 = _mm256_sub_pd(z89, u10_11);

    StockhamQuadFirstPairs64 {
        p01,
        p45,
        p89,
        p12_13,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stockham_quad_groups_eight64_second_pairs(
    src_ptr: *const Complex64,
    first_ptr: *const Complex64,
    second_ptr: *const Complex64,
    third_ptr: *const Complex64,
    radix: usize,
    j: usize,
) -> StockhamQuadSecondPairs64 {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_sub_pd};

    let w1 = *first_ptr.add(j);
    let w1r = _mm256_set1_pd(w1.re);
    let w1i = _mm256_set1_pd(w1.im);

    let src_base = j << 4;
    let x01 = _mm256_loadu_pd(src_ptr.add(src_base).cast::<f64>());
    let x23 = _mm256_loadu_pd(src_ptr.add(src_base + 2).cast::<f64>());
    let x45 = _mm256_loadu_pd(src_ptr.add(src_base + 4).cast::<f64>());
    let x67 = _mm256_loadu_pd(src_ptr.add(src_base + 6).cast::<f64>());
    let x89 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 8).cast::<f64>()),
    );
    let x10_11 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 10).cast::<f64>()),
    );
    let x12_13 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 12).cast::<f64>()),
    );
    let x14_15 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 14).cast::<f64>()),
    );

    let y89 = _mm256_sub_pd(x01, x89);
    let y10_11 = _mm256_sub_pd(x23, x10_11);
    let y12_13 = _mm256_sub_pd(x45, x12_13);
    let y14_15 = _mm256_sub_pd(x67, x14_15);

    let w21 = *second_ptr.add(j + radix);
    let w21r = _mm256_set1_pd(w21.re);
    let w21i = _mm256_set1_pd(w21.im);
    let t12_13 = cmul_vec64(w21r, w21i, y12_13);
    let t14_15 = cmul_vec64(w21r, w21i, y14_15);
    let z45 = _mm256_add_pd(y89, t12_13);
    let z67 = _mm256_add_pd(y10_11, t14_15);
    let z12_13 = _mm256_sub_pd(y89, t12_13);
    let z14_15 = _mm256_sub_pd(y10_11, t14_15);

    let w31 = *third_ptr.add(j + radix);
    let w31r = _mm256_set1_pd(w31.re);
    let w31i = _mm256_set1_pd(w31.im);
    let u67 = cmul_vec64(w31r, w31i, z67);
    let p23 = _mm256_add_pd(z45, u67);
    let p10_11 = _mm256_sub_pd(z45, u67);

    let w33 = *third_ptr.add(j + 3 * radix);
    let w33r = _mm256_set1_pd(w33.re);
    let w33i = _mm256_set1_pd(w33.im);
    let u14_15 = cmul_vec64(w33r, w33i, z14_15);
    let p67 = _mm256_add_pd(z12_13, u14_15);
    let p14_15 = _mm256_sub_pd(z12_13, u14_15);

    StockhamQuadSecondPairs64 {
        p23,
        p67,
        p10_11,
        p14_15,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stockham_quad_split_pair32(
    value: std::arch::x86_64::__m256,
    twiddle: Complex32,
) -> [std::arch::x86_64::__m128; 2] {
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_set1_ps, _mm_sub_ps,
    };

    let low = _mm256_castps256_ps128(value);
    let high = _mm256_extractf128_ps(value, 1);
    let product = cmul_pair32(_mm_set1_ps(twiddle.re), _mm_set1_ps(twiddle.im), high);
    [_mm_add_ps(low, product), _mm_sub_ps(low, product)]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stockham_quad_store_pair32(
    dst_ptr: *mut Complex32,
    low_index: usize,
    high_index: usize,
    pair: std::arch::x86_64::__m128,
    twiddle: Complex32,
) {
    use std::arch::x86_64::{_mm_add_ps, _mm_movehl_ps, _mm_set1_ps, _mm_sub_ps};

    let odd = _mm_movehl_ps(pair, pair);
    let product = cmul_pair32(_mm_set1_ps(twiddle.re), _mm_set1_ps(twiddle.im), odd);
    store_complex32_low(dst_ptr.add(low_index), _mm_add_ps(pair, product));
    store_complex32_low(dst_ptr.add(high_index), _mm_sub_ps(pair, product));
}

/// AVX/FMA radix-16 Stockham suffix for f64 when `groups == 8`.
///
/// The leaf evaluates four radix-2 Stockham stages for one digit `j` at a time.
/// The first three stages produce eight YMM pairs; the fourth stage consumes
/// each pair immediately through `stockham_quad_store_pair64`. This preserves
/// Stockham autosort order without a bit-reversal permutation and avoids
/// pairing adjacent digits, which measured as high register pressure in the
/// rejected contiguous-store candidate.
///
/// Algebraically, each stored band is `a ± W*b`, where `a` and `b` are the even
/// and odd outputs of the first three Stockham stages for the same digit. This
/// is exactly the fourth radix-2 Stockham recurrence, so direct substitution
/// proves equality with four scalar Stockham stages.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stockham_quad_groups_eight64_low_live(
    src: &[Complex64],
    dst: &mut [Complex64],
    radix: usize,
    first_twiddles: &[Complex64],
    second_twiddles: &[Complex64],
    third_twiddles: &[Complex64],
    fourth_twiddles: &[Complex64],
) {
    let n = src.len();
    debug_assert_eq!(n, radix << 4);
    debug_assert_eq!(dst.len(), n);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= radix << 1);
    debug_assert!(third_twiddles.len() >= radix << 2);
    debug_assert!(fourth_twiddles.len() >= radix << 3);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();
    let fourth_ptr = fourth_twiddles.as_ptr();

    for j in 0..radix {
        let first = stockham_quad_groups_eight64_first_pairs(
            src_ptr, first_ptr, second_ptr, third_ptr, radix, j,
        );
        stockham_quad_store_pair64(dst_ptr, j, j + 8 * radix, first.p01, *fourth_ptr.add(j));
        stockham_quad_store_pair64(
            dst_ptr,
            j + 2 * radix,
            j + 10 * radix,
            first.p45,
            *fourth_ptr.add(j + 2 * radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 4 * radix,
            j + 12 * radix,
            first.p89,
            *fourth_ptr.add(j + 4 * radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 6 * radix,
            j + 14 * radix,
            first.p12_13,
            *fourth_ptr.add(j + 6 * radix),
        );

        let second = stockham_quad_groups_eight64_second_pairs(
            src_ptr, first_ptr, second_ptr, third_ptr, radix, j,
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + radix,
            j + 9 * radix,
            second.p23,
            *fourth_ptr.add(j + radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 3 * radix,
            j + 11 * radix,
            second.p67,
            *fourth_ptr.add(j + 3 * radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 5 * radix,
            j + 13 * radix,
            second.p10_11,
            *fourth_ptr.add(j + 5 * radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 7 * radix,
            j + 15 * radix,
            second.p14_15,
            *fourth_ptr.add(j + 7 * radix),
        );
    }
}

/// AVX/FMA radix-16 Stockham suffix for f32 when `groups == 8`.
///
/// The leaf evaluates the same four scalar Stockham recurrences as
/// `stage_quad32`. The first two stages operate on four contiguous YMM loads;
/// the third stage splits each YMM into two XMM complex pairs, and the final
/// stage stores each pair directly into its Stockham output band. This keeps
/// the live set bounded by the 4x4 local radix-16 shape and avoids heap or
/// stack tiles.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn stockham_quad_groups_eight32(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
    third_twiddles: &[Complex32],
    fourth_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_sub_ps};

    let n = src.len();
    debug_assert_eq!(n, radix << 4);
    debug_assert_eq!(dst.len(), n);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= radix << 1);
    debug_assert!(third_twiddles.len() >= radix << 2);
    debug_assert!(fourth_twiddles.len() >= radix << 3);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();
    let fourth_ptr = fourth_twiddles.as_ptr();

    for j in 0..radix {
        let w1 = *first_ptr.add(j);
        let w20 = *second_ptr.add(j);
        let w21 = *second_ptr.add(j + radix);

        let w1r = _mm256_set1_ps(w1.re);
        let w1i = _mm256_set1_ps(w1.im);
        let w20r = _mm256_set1_ps(w20.re);
        let w20i = _mm256_set1_ps(w20.im);
        let w21r = _mm256_set1_ps(w21.re);
        let w21i = _mm256_set1_ps(w21.im);

        let src_base = j << 4;
        let x0123 = _mm256_loadu_ps(src_ptr.add(src_base).cast::<f32>());
        let x4567 = _mm256_loadu_ps(src_ptr.add(src_base + 4).cast::<f32>());
        let x89ab = cmul_vec32(
            w1r,
            w1i,
            _mm256_loadu_ps(src_ptr.add(src_base + 8).cast::<f32>()),
        );
        let xcdef = cmul_vec32(
            w1r,
            w1i,
            _mm256_loadu_ps(src_ptr.add(src_base + 12).cast::<f32>()),
        );

        let y0123 = _mm256_add_ps(x0123, x89ab);
        let y4567 = _mm256_add_ps(x4567, xcdef);
        let y89ab = _mm256_sub_ps(x0123, x89ab);
        let ycdef = _mm256_sub_ps(x4567, xcdef);

        let t4567 = cmul_vec32(w20r, w20i, y4567);
        let tcdef = cmul_vec32(w21r, w21i, ycdef);
        let z0123 = _mm256_add_ps(y0123, t4567);
        let z4567 = _mm256_add_ps(y89ab, tcdef);
        let z89ab = _mm256_sub_ps(y0123, t4567);
        let zcdef = _mm256_sub_ps(y89ab, tcdef);

        let [p01, p89] = stockham_quad_split_pair32(z0123, *third_ptr.add(j));
        let [p23, p10_11] = stockham_quad_split_pair32(z4567, *third_ptr.add(j + radix));
        let [p45, p12_13] = stockham_quad_split_pair32(z89ab, *third_ptr.add(j + 2 * radix));
        let [p67, p14_15] = stockham_quad_split_pair32(zcdef, *third_ptr.add(j + 3 * radix));

        stockham_quad_store_pair32(dst_ptr, j, j + 8 * radix, p01, *fourth_ptr.add(j));
        stockham_quad_store_pair32(
            dst_ptr,
            j + radix,
            j + 9 * radix,
            p23,
            *fourth_ptr.add(j + radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 2 * radix,
            j + 10 * radix,
            p45,
            *fourth_ptr.add(j + 2 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 3 * radix,
            j + 11 * radix,
            p67,
            *fourth_ptr.add(j + 3 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 4 * radix,
            j + 12 * radix,
            p89,
            *fourth_ptr.add(j + 4 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 5 * radix,
            j + 13 * radix,
            p10_11,
            *fourth_ptr.add(j + 5 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 6 * radix,
            j + 14 * radix,
            p12_13,
            *fourth_ptr.add(j + 6 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 7 * radix,
            j + 15 * radix,
            p14_15,
            *fourth_ptr.add(j + 7 * radix),
        );
    }
}

/// Type-selected Stockham kernel policy.
///
/// The transform schedule is generic over this trait, while SIMD leaves remain
/// concrete `Complex32`/`Complex64` functions. Rust monomorphizes each
/// precision into a separate loop body, so `MAX_FUSED_STAGES` removes inactive
/// fused codelet branches at compile time without runtime type inspection,
/// vtables, allocation, or runtime precision switches.
trait StockhamPrecision: private::Sealed {
    type Real: Copy;
    type Complex: Copy;

    const MAX_FUSED_STAGES: u32;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let _ = (stride, n, input_is_data);
        true
    }

    #[inline]
    fn stage_quad_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let _ = (stride, n, input_is_data);
        false
    }

    fn stage(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        twiddles: &[Self::Complex],
    );

    fn stage_pair(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
    );
    fn stage_triple(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
    );
    fn stage_quad(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
        fourth_twiddles: &[Self::Complex],
    );

    fn scale(data: &mut [Self::Complex], scale: Self::Real);
}

#[cfg(target_arch = "x86_64")]
trait StockhamRadix16AvxLeaf: StockhamPrecision {
    unsafe fn stage_quad_groups_eight_avx_fma(
        src: &[Self::Complex],
        dst: &mut [Self::Complex],
        radix: usize,
        first_twiddles: &[Self::Complex],
        second_twiddles: &[Self::Complex],
        third_twiddles: &[Self::Complex],
        fourth_twiddles: &[Self::Complex],
    );
}

/// Compile-time Stockham fusion-width policy.
///
/// The word `radix` has two distinct meanings in FFT code:
///
/// - factor radix: a DFT factor such as 3, 5, or 7 in a mixed-radix plan;
/// - fused radix-2 width: the number of adjacent radix-2 Stockham stages folded
///   into one autosort codelet.
///
/// This trait encodes the second meaning. `FUSED_RADIX2_WIDTH = 2^STAGE_COUNT`
/// is a codelet width, not a transform factorization choice. The runtime value
/// passed to `apply` is the current Stockham stage stride.
trait StockhamFusion: private::Sealed {
    const STAGE_COUNT: u32;
    const FUSED_RADIX2_WIDTH: usize;
    const TWIDDLE_STRIDE_FACTOR: usize;
}

impl StockhamFusion for StockhamFused1 {
    const STAGE_COUNT: u32 = 1;
    const FUSED_RADIX2_WIDTH: usize = 2;
    const TWIDDLE_STRIDE_FACTOR: usize = 1;
}

impl StockhamFusion for StockhamFused2 {
    const STAGE_COUNT: u32 = 2;
    const FUSED_RADIX2_WIDTH: usize = 4;
    const TWIDDLE_STRIDE_FACTOR: usize = 3;
}

impl StockhamFusion for StockhamFused3 {
    const STAGE_COUNT: u32 = 3;
    const FUSED_RADIX2_WIDTH: usize = 8;
    const TWIDDLE_STRIDE_FACTOR: usize = 7;
}

impl StockhamFusion for StockhamFused4 {
    const STAGE_COUNT: u32 = 4;
    const FUSED_RADIX2_WIDTH: usize = 16;
    const TWIDDLE_STRIDE_FACTOR: usize = 15;
}
#[inline]
fn fusion_fits<C: StockhamFusion>(stride: usize, n: usize) -> bool {
    stride <= n / C::FUSED_RADIX2_WIDTH
}

#[inline]
fn fusion_twiddle_len<C: StockhamFusion>(stride: usize) -> usize {
    stride * C::TWIDDLE_STRIDE_FACTOR
}

#[inline]
fn stockham_twiddle_table_len(n: usize) -> usize {
    debug_assert!(n >= 2);
    n - 1
}

struct StockhamTwiddleCursor<'a, T> {
    ptr: *const T,
    len: usize,
    consumed: usize,
    _lifetime: std::marker::PhantomData<&'a [T]>,
}

impl<'a, T> StockhamTwiddleCursor<'a, T> {
    #[inline]
    fn new(twiddles: &'a [T]) -> Self {
        Self {
            ptr: twiddles.as_ptr(),
            len: twiddles.len(),
            consumed: 0,
            _lifetime: std::marker::PhantomData,
        }
    }

    #[inline]
    unsafe fn take(&mut self, len: usize) -> &'a [T] {
        debug_assert!(self.consumed + len <= self.len);
        let start = self.consumed;
        self.consumed += len;
        unsafe { std::slice::from_raw_parts(self.ptr.add(start), len) }
    }

    #[inline]
    fn consumed(&self) -> usize {
        self.consumed
    }
}

#[inline]
unsafe fn stockham_twiddle_subslice<T>(twiddles: &[T], start: usize, len: usize) -> &[T] {
    debug_assert!(start + len <= twiddles.len());
    unsafe { std::slice::from_raw_parts(twiddles.as_ptr().add(start), len) }
}

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl StockhamPrecision for F64Stockham {
    type Real = f64;
    type Complex = Complex64;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage(src: &[Complex64], dst: &mut [Complex64], radix: usize, twiddles: &[Complex64]) {
        stage_impl(src, dst, radix, twiddles);
    }

    #[inline]
    fn stage_pair(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
    ) {
        stage_pair_impl(src, dst, radix, first_twiddles, second_twiddles);
    }

    #[inline]
    fn stage_triple(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) {
        stage_triple_impl(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
        );
    }

    #[inline]
    fn stage_quad(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
        fourth_twiddles: &[Complex64],
    ) {
        stage_quad_impl(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
            fourth_twiddles,
        );
    }

    #[inline]
    fn scale(data: &mut [Complex64], scale: f64) {
        normalize_inplace_c64(data, scale);
    }
}

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl StockhamPrecision for F32Stockham {
    type Real = f32;
    type Complex = Complex32;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let _ = (stride, n, input_is_data);
        false
    }

    #[inline]
    fn stage(src: &[Complex32], dst: &mut [Complex32], radix: usize, twiddles: &[Complex32]) {
        stage_impl(src, dst, radix, twiddles);
    }

    #[inline]
    fn stage_pair(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
    ) {
        stage_pair_impl(src, dst, radix, first_twiddles, second_twiddles);
    }

    #[inline]
    fn stage_triple(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
    ) {
        stage_triple_impl(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
        );
    }

    #[inline]
    fn stage_quad(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
        fourth_twiddles: &[Complex32],
    ) {
        stage_quad_impl(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
            fourth_twiddles,
        );
    }

    #[inline]
    fn scale(data: &mut [Complex32], scale: f32) {
        normalize_inplace_c32(data, scale);
    }
}

#[cfg(target_arch = "x86_64")]
struct F64StockhamAvxFma;

#[cfg(target_arch = "x86_64")]
impl private::Sealed for F64StockhamAvxFma {}

#[cfg(target_arch = "x86_64")]
impl StockhamRadix16AvxLeaf for F64StockhamAvxFma {
    #[inline]
    unsafe fn stage_quad_groups_eight_avx_fma(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
        fourth_twiddles: &[Complex64],
    ) {
        unsafe {
            stockham_quad_groups_eight64_low_live(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
                fourth_twiddles,
            )
        };
    }
}

#[cfg(target_arch = "x86_64")]
impl StockhamPrecision for F64StockhamAvxFma {
    type Real = f64;
    type Complex = Complex64;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        // `groups == 4` means exactly three stages remain and is a zero-copy
        // win only when the source is scratch, because the radix-8 autosort
        // suffix then writes the final ping-pong pass into `data`.
        // `groups > 4` leaves at least one additional pass after the radix-8
        // stage; the fused stage still reduces arithmetic scheduling overhead
        // without changing the final ping-pong parity.
        let groups = n / (stride << 1);
        groups > 4 || (groups == 4 && !input_is_data)
    }

    #[inline]
    fn stage_quad_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let _ = (stride, n, input_is_data);
        false
    }

    #[inline]
    fn stage(src: &[Complex64], dst: &mut [Complex64], radix: usize, twiddles: &[Complex64]) {
        let groups = src.len() / (radix << 1);
        if groups == 1 && radix >= 2 {
            unsafe { stage64_groups_one_avx_fma(src, dst, radix, twiddles) };
        } else if groups >= 2 {
            unsafe { stage64_avx_fma(src, dst, radix, twiddles) };
        } else {
            stage_impl(src, dst, radix, twiddles);
        }
    }

    #[inline]
    fn stage_pair(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 {
            if src.len() >= 8 {
                unsafe { stage_pair64_radix1_avx_fma(src, dst, second_twiddles) };
            } else {
                stage_pair_impl(src, dst, radix, first_twiddles, second_twiddles);
            }
        } else if groups == 2 && radix >= 2 {
            unsafe {
                stage_pair64_groups_two_avx_fma(src, dst, radix, first_twiddles, second_twiddles)
            };
        } else if groups >= 4 {
            unsafe { stage_pair64_avx_fma(src, dst, radix, first_twiddles, second_twiddles) };
        } else {
            stage_pair_impl(src, dst, radix, first_twiddles, second_twiddles);
        }
    }

    #[inline]
    fn stage_triple(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 && groups >= 8 && stockham_f64_stage_is_l1_resident(src.len()) {
            unsafe { stage_triple64_radix1_avx_fma(src, dst, second_twiddles, third_twiddles) };
        } else if groups == 8 {
            unsafe {
                stage_triple64_groups_eight_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                )
            };
        } else if groups >= 8 {
            if stockham_f64_stage_is_l1_resident(src.len()) {
                unsafe {
                    stage_triple64_low_live_avx_fma(
                        src,
                        dst,
                        radix,
                        groups,
                        first_twiddles,
                        second_twiddles,
                        third_twiddles,
                    )
                };
            } else {
                unsafe {
                    stage_triple64_throughput_avx_fma(
                        src,
                        dst,
                        radix,
                        groups,
                        first_twiddles,
                        second_twiddles,
                        third_twiddles,
                    )
                };
            }
        } else if groups == 4 {
            unsafe {
                stage_triple64_quarter_groups_one_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                )
            };
        } else {
            stage_triple_impl(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
            );
        }
    }

    #[inline]
    fn stage_quad(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
        fourth_twiddles: &[Complex64],
    ) {
        let groups = src.len() / (radix << 1);
        if groups == 8 {
            unsafe {
                <Self as StockhamRadix16AvxLeaf>::stage_quad_groups_eight_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                    fourth_twiddles,
                )
            };
        } else {
            stage_quad_impl(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
                fourth_twiddles,
            );
        }
    }

    fn scale(data: &mut [Complex64], scale: f64) {
        normalize_inplace_c64(data, scale);
    }
}

#[cfg(target_arch = "x86_64")]
struct F32StockhamAvxFma;

#[cfg(target_arch = "x86_64")]
impl private::Sealed for F32StockhamAvxFma {}

#[cfg(target_arch = "x86_64")]
impl StockhamRadix16AvxLeaf for F32StockhamAvxFma {
    #[inline]
    unsafe fn stage_quad_groups_eight_avx_fma(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
        fourth_twiddles: &[Complex32],
    ) {
        unsafe {
            stockham_quad_groups_eight32(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
                fourth_twiddles,
            )
        };
    }
}

#[cfg(target_arch = "x86_64")]
impl StockhamPrecision for F32StockhamAvxFma {
    type Real = f32;
    type Complex = Complex32;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let groups = n / (stride << 1);
        groups > 4 || (groups == 4 && !input_is_data)
    }

    #[inline]
    fn stage_quad_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let groups = n / (stride << 1);
        let _ = (stride, groups, n, input_is_data);
        false
    }

    #[inline]
    fn stage(src: &[Complex32], dst: &mut [Complex32], radix: usize, twiddles: &[Complex32]) {
        let groups = src.len() / (radix << 1);
        if groups == 1 && radix >= 2 {
            unsafe { stage32_groups_one_avx_fma(src, dst, radix, twiddles) };
        } else if groups >= 4 {
            unsafe { stage32_avx_fma(src, dst, radix, twiddles) };
        } else {
            stage_impl(src, dst, radix, twiddles);
        }
    }

    #[inline]
    fn stage_pair(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 {
            if src.len() >= 16 {
                unsafe { stage_pair32_radix1_avx_fma(src, dst, second_twiddles) };
            } else {
                stage_pair_impl(src, dst, radix, first_twiddles, second_twiddles);
            }
        } else if groups >= 8 {
            unsafe { stage_pair32_avx_fma(src, dst, radix, first_twiddles, second_twiddles) };
        } else if groups == 4 {
            unsafe {
                stage_pair32_quarter_groups_two_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                )
            };
        } else if groups == 2 {
            unsafe {
                stage_pair32_groups_two_avx_fma(src, dst, radix, first_twiddles, second_twiddles)
            };
        } else {
            stage_pair_impl(src, dst, radix, first_twiddles, second_twiddles);
        }
    }

    #[inline]
    fn stage_triple(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 && groups >= 8 && src.len() >= 32 {
            unsafe { stage_triple32_radix1_avx_fma(src, dst, second_twiddles, third_twiddles) };
        } else if groups >= 16 {
            if stockham_f32_stage_is_l1_resident(src.len()) {
                unsafe {
                    stage_triple32_low_live_avx_fma(
                        src,
                        dst,
                        radix,
                        groups,
                        first_twiddles,
                        second_twiddles,
                        third_twiddles,
                    )
                };
            } else {
                unsafe {
                    stage_triple32_avx_fma(
                        src,
                        dst,
                        radix,
                        groups,
                        first_twiddles,
                        second_twiddles,
                        third_twiddles,
                    )
                };
            }
        } else if groups == 8 {
            unsafe {
                stage_triple32_quarter_groups_two_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                )
            };
        } else if groups == 4 {
            unsafe {
                stage_triple32_quarter_groups_one_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                )
            };
        } else {
            stage_triple_impl(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
            );
        }
    }

    #[inline]
    fn stage_quad(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
        fourth_twiddles: &[Complex32],
    ) {
        let groups = src.len() / (radix << 1);
        if groups == 8 {
            unsafe {
                <Self as StockhamRadix16AvxLeaf>::stage_quad_groups_eight_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                    fourth_twiddles,
                )
            };
        } else {
            stage_quad_impl(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
                fourth_twiddles,
            );
        }
    }

    fn scale(data: &mut [Complex32], scale: f32) {
        normalize_inplace_c32(data, scale);
    }
}

#[inline]
fn transform<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
    scale: Option<P::Real>,
) {
    let n = data.len();
    if n <= 1 {
        if let Some(scale) = scale {
            P::scale(data, scale);
        }
        return;
    }
    debug_assert!(n.is_power_of_two());
    debug_assert!(twiddles.len() >= stockham_twiddle_table_len(n));
    let mut cursor = StockhamTwiddleCursor::new(twiddles);
    let mut stride = 1usize;
    let mut input_is_data = true;
    while stride < n {
        if P::MAX_FUSED_STAGES >= StockhamFused4::STAGE_COUNT
            && fusion_fits::<StockhamFused4>(stride, n)
            && P::stage_quad_enabled(stride, n, input_is_data)
        {
            let twiddle_len = fusion_twiddle_len::<StockhamFused4>(stride);
            let fusion_twiddles = unsafe { cursor.take(twiddle_len) };
            let first_twiddles = unsafe { stockham_twiddle_subslice(fusion_twiddles, 0, stride) };
            let second_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride, stride << 1) };
            let third_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride * 3, stride << 2) };
            let fourth_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride * 7, stride << 3) };
            if input_is_data {
                P::stage_quad(
                    data,
                    scratch,
                    stride,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                    fourth_twiddles,
                );
            } else {
                P::stage_quad(
                    scratch,
                    data,
                    stride,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                    fourth_twiddles,
                );
            }
            input_is_data = !input_is_data;
            stride <<= StockhamFused4::STAGE_COUNT;
        } else if P::MAX_FUSED_STAGES >= StockhamFused3::STAGE_COUNT
            && fusion_fits::<StockhamFused3>(stride, n)
            && P::stage_triple_enabled(stride, n, input_is_data)
        {
            let twiddle_len = fusion_twiddle_len::<StockhamFused3>(stride);
            let fusion_twiddles = unsafe { cursor.take(twiddle_len) };
            let first_twiddles = unsafe { stockham_twiddle_subslice(fusion_twiddles, 0, stride) };
            let second_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride, stride << 1) };
            let third_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride * 3, stride << 2) };
            if input_is_data {
                P::stage_triple(
                    data,
                    scratch,
                    stride,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                );
            } else {
                P::stage_triple(
                    scratch,
                    data,
                    stride,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                );
            }
            input_is_data = !input_is_data;
            stride <<= StockhamFused3::STAGE_COUNT;
        } else if P::MAX_FUSED_STAGES >= StockhamFused2::STAGE_COUNT
            && fusion_fits::<StockhamFused2>(stride, n)
        {
            let twiddle_len = fusion_twiddle_len::<StockhamFused2>(stride);
            let fusion_twiddles = unsafe { cursor.take(twiddle_len) };
            let first_twiddles = unsafe { stockham_twiddle_subslice(fusion_twiddles, 0, stride) };
            let second_twiddles =
                unsafe { stockham_twiddle_subslice(fusion_twiddles, stride, stride << 1) };
            if input_is_data {
                P::stage_pair(data, scratch, stride, first_twiddles, second_twiddles);
            } else {
                P::stage_pair(scratch, data, stride, first_twiddles, second_twiddles);
            }
            input_is_data = !input_is_data;
            stride <<= StockhamFused2::STAGE_COUNT;
        } else {
            let twiddle_len = fusion_twiddle_len::<StockhamFused1>(stride);
            let stage_twiddles = unsafe { cursor.take(twiddle_len) };
            if input_is_data {
                P::stage(data, scratch, stride, stage_twiddles);
            } else {
                P::stage(scratch, data, stride, stage_twiddles);
            }
            input_is_data = !input_is_data;
            stride <<= StockhamFused1::STAGE_COUNT;
        }
    }
    debug_assert_eq!(cursor.consumed(), stockham_twiddle_table_len(n));
    if !input_is_data {
        data.copy_from_slice(scratch);
    }
    if let Some(scale) = scale {
        P::scale(data, scale);
    }
}

#[inline]
fn transform_len4096_four_triples<P: StockhamPrecision>(
    data: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
) {
    debug_assert_eq!(data.len(), 4096);
    debug_assert_eq!(scratch.len(), 4096);
    debug_assert!(twiddles.len() >= 4095);

    P::stage_triple(
        data,
        scratch,
        1,
        &twiddles[0..1],
        &twiddles[1..3],
        &twiddles[3..7],
    );
    P::stage_triple(
        scratch,
        data,
        8,
        &twiddles[7..15],
        &twiddles[15..31],
        &twiddles[31..63],
    );
    P::stage_triple(
        data,
        scratch,
        64,
        &twiddles[63..127],
        &twiddles[127..255],
        &twiddles[255..511],
    );
    P::stage_triple(
        scratch,
        data,
        512,
        &twiddles[511..1023],
        &twiddles[1023..2047],
        &twiddles[2047..4095],
    );
}

#[cfg(all(test, target_arch = "x86_64"))]
#[inline]
fn stockham_mixed_twiddle_64<const ROWS: usize, const COLS: usize>(
    twiddles: &[Complex64],
    row: usize,
    col: usize,
) -> Complex64 {
    let exponent = row * col;
    if exponent == 0 {
        return Complex64::new(1.0, 0.0);
    }
    let half_n = (ROWS * COLS) >> 1;
    let stage_base = half_n - 1;
    if exponent < half_n {
        twiddles[stage_base + exponent]
    } else {
        -twiddles[stage_base + exponent - half_n]
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[inline]
fn stockham_mixed_twiddle_32<const ROWS: usize, const COLS: usize>(
    twiddles: &[Complex32],
    row: usize,
    col: usize,
) -> Complex32 {
    let exponent = row * col;
    if exponent == 0 {
        return Complex32::new(1.0, 0.0);
    }
    let half_n = (ROWS * COLS) >> 1;
    let stage_base = half_n - 1;
    if exponent < half_n {
        twiddles[stage_base + exponent]
    } else {
        -twiddles[stage_base + exponent - half_n]
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
type PackedF32x4 = [Complex32; 4];

#[cfg(all(test, target_arch = "x86_64"))]
type PackedF64x2 = [Complex64; 2];

#[cfg(all(test, target_arch = "x86_64"))]
fn build_butterfly512_twiddles_32(twiddles: &[Complex32]) -> [PackedF32x4; 120] {
    debug_assert!(twiddles.len() >= 511);
    core::array::from_fn(|index| {
        let row = (index % 15) + 1;
        let col_base = (index / 15) * 4;
        core::array::from_fn(|lane| {
            stockham_mixed_twiddle_32::<16, 32>(twiddles, row, col_base + lane)
        })
    })
}

#[cfg(all(test, target_arch = "x86_64"))]
fn build_butterfly512_twiddles_64(twiddles: &[Complex64]) -> [PackedF64x2; 240] {
    debug_assert!(twiddles.len() >= 511);
    core::array::from_fn(|index| {
        let row = (index % 15) + 1;
        let col_base = (index / 15) * 2;
        core::array::from_fn(|lane| {
            stockham_mixed_twiddle_64::<16, 32>(twiddles, row, col_base + lane)
        })
    })
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn hybrid_radix8x512_64_avx_fma<const INVERSE: bool>(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    const ROWS: usize = 8;
    const COLS: usize = 512;
    debug_assert_eq!(data.len(), ROWS * COLS);
    debug_assert_eq!(scratch.len(), ROWS * COLS);
    debug_assert!(twiddles.len() >= ROWS * COLS - 1);

    let row_twiddles = &twiddles[..COLS - 1];
    stage_triple64_radix1_avx_fma(data, scratch, &twiddles[1..3], &twiddles[3..7]);
    let mut r = 1;
    while r < ROWS {
        let row_base = r * COLS;
        let mut c = 1;
        while c < COLS {
            scratch[row_base + c] *= stockham_mixed_twiddle_64::<ROWS, COLS>(twiddles, r, c);
            c += 1;
        }
        r += 1;
    }

    for r in 0..ROWS {
        let row = &mut scratch[r * COLS..(r + 1) * COLS];
        let row_scratch = &mut data[r * COLS..(r + 1) * COLS];
        fixed_len512_avx_fma(row, row_scratch, row_twiddles);
    }

    for r in 0..ROWS {
        for c in 0..COLS {
            data[c * ROWS + r] = scratch[r * COLS + c];
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn hybrid_radix8x512_32_avx_fma<const INVERSE: bool>(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    const ROWS: usize = 8;
    const COLS: usize = 512;
    debug_assert_eq!(data.len(), ROWS * COLS);
    debug_assert_eq!(scratch.len(), ROWS * COLS);
    debug_assert!(twiddles.len() >= ROWS * COLS - 1);

    let row_twiddles = &twiddles[..COLS - 1];
    stage_triple32_radix1_avx_fma(data, scratch, &twiddles[1..3], &twiddles[3..7]);
    let mut r = 1;
    while r < ROWS {
        let row_base = r * COLS;
        let mut c = 1;
        while c < COLS {
            scratch[row_base + c] *= stockham_mixed_twiddle_32::<ROWS, COLS>(twiddles, r, c);
            c += 1;
        }
        r += 1;
    }

    for r in 0..ROWS {
        let row = &mut scratch[r * COLS..(r + 1) * COLS];
        let row_scratch = &mut data[r * COLS..(r + 1) * COLS];
        fixed_len512_32_avx_fma(row, row_scratch, row_twiddles);
    }

    for r in 0..ROWS {
        for c in 0..COLS {
            data[c * ROWS + r] = scratch[r * COLS + c];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn forward64_avx_with_scratch(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    if data.len() == 64 {
        fixed_len64_avx_fma(data, scratch, twiddles);
        return;
    }
    if data.len() == 4096 && twiddles.get(1).is_some_and(|w| w.im < 0.0) {
        transform_len4096_four_triples::<F64StockhamAvxFma>(data, scratch, twiddles);
        return;
    }
    transform::<F64StockhamAvxFma>(data, scratch, twiddles, None);
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len512_avx_fma(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;
    // Pass 1: radix 1, groups 256
    for j in 0..1 {
        let w = twiddles[j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 512;
        let dst_base = j * 256;
        let mut k = 0;
        while k < 256 {
            let x0 = _mm256_loadu_pd(data.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 256 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(),
                t,
            );
            k += 2;
        }
    }
    // Pass 2: radix 2, groups 128
    for j in 0..2 {
        let w = twiddles[1 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 256;
        let dst_base = j * 128;
        let mut k = 0;
        while k < 128 {
            let x0 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + 128 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(), t);
            k += 2;
        }
    }
    // Pass 3: radix 4, groups 64
    for j in 0..4 {
        let w = twiddles[3 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 128;
        let dst_base = j * 64;
        let mut k = 0;
        while k < 64 {
            let x0 = _mm256_loadu_pd(data.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 64 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(),
                t,
            );
            k += 2;
        }
    }
    // Pass 4: radix 8, groups 32
    for j in 0..8 {
        let w = twiddles[7 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 64;
        let dst_base = j * 32;
        let mut k = 0;
        while k < 32 {
            let x0 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + 32 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(), t);
            k += 2;
        }
    }
    // Pass 5: radix 16, groups 16
    for j in 0..16 {
        let w = twiddles[15 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 32;
        let dst_base = j * 16;
        let mut k = 0;
        while k < 16 {
            let x0 = _mm256_loadu_pd(data.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 16 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(),
                t,
            );
            k += 2;
        }
    }
    // Pass 6: radix 32, groups 8
    for j in 0..32 {
        let w = twiddles[31 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 16;
        let dst_base = j * 8;
        let mut k = 0;
        while k < 8 {
            let x0 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + 8 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(), t);
            k += 2;
        }
    }
    // Pass 7: radix 64, groups 4
    for j in 0..64 {
        let w = twiddles[63 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 8;
        let dst_base = j * 4;
        let mut k = 0;
        while k < 4 {
            let x0 = _mm256_loadu_pd(data.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 4 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(),
                t,
            );
            k += 2;
        }
    }
    // Pass 8: radix 128, groups 2
    for j in 0..128 {
        let w = twiddles[127 + j];
        let w_re = _mm256_set1_pd(w.re);
        let w_im = _mm256_set1_pd(w.im);
        let src_base = j * 4;
        let dst_base = j * 2;
        let mut k = 0;
        while k < 2 {
            let x0 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + k).cast::<f64>());
            let x1 = _mm256_loadu_pd(scratch.as_ptr().add(src_base + 2 + k).cast::<f64>());
            let x1_shuf = _mm256_permute_pd(x1, 0b0101);
            let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
            let s = _mm256_add_pd(x0, product);
            let t = _mm256_sub_pd(x0, product);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + k).cast::<f64>(), s);
            _mm256_storeu_pd(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f64>(), t);
            k += 2;
        }
    }
    // Pass 9: radix 256, groups 1
    let mut j = 0;
    while j < 256 {
        let w0 = twiddles[255 + j];
        let w1 = twiddles[255 + j + 1];
        let w_re = _mm256_set_pd(w1.re, w1.re, w0.re, w0.re);
        let w_im = _mm256_set_pd(w1.im, w1.im, w0.im, w0.im);
        let src_base = j * 2;
        let dst_base = j;
        let d0 = _mm256_loadu_pd(data.as_ptr().add(src_base).cast::<f64>());
        let d1 = _mm256_loadu_pd(data.as_ptr().add(src_base + 2).cast::<f64>());
        let x0 = _mm256_permute2f128_pd(d0, d1, 0x20);
        let x1 = _mm256_permute2f128_pd(d0, d1, 0x31);
        let x1_shuf = _mm256_permute_pd(x1, 0b0101);
        let product = _mm256_addsub_pd(_mm256_mul_pd(w_re, x1), _mm256_mul_pd(w_im, x1_shuf));
        let s = _mm256_add_pd(x0, product);
        let t = _mm256_sub_pd(x0, product);
        _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base).cast::<f64>(), s);
        _mm256_storeu_pd(scratch.as_mut_ptr().add(dst_base + 256).cast::<f64>(), t);
        j += 2;
    }
    data.copy_from_slice(scratch);
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len512_32_avx_fma(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;
    // Pass 1: radix 1, groups 256
    for j in 0..1 {
        let w = twiddles[j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 512;
        let dst_base = j * 256;
        let mut k = 0;
        while k < 256 {
            let x0 = _mm256_loadu_ps(data.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(data.as_ptr().add(src_base + 256 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(),
                t,
            );
            k += 4;
        }
    }
    // Pass 2: radix 2, groups 128
    for j in 0..2 {
        let w = twiddles[1 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 256;
        let dst_base = j * 128;
        let mut k = 0;
        while k < 128 {
            let x0 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + 128 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(), t);
            k += 4;
        }
    }
    // Pass 3: radix 4, groups 64
    for j in 0..4 {
        let w = twiddles[3 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 128;
        let dst_base = j * 64;
        let mut k = 0;
        while k < 64 {
            let x0 = _mm256_loadu_ps(data.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(data.as_ptr().add(src_base + 64 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(),
                t,
            );
            k += 4;
        }
    }
    // Pass 4: radix 8, groups 32
    for j in 0..8 {
        let w = twiddles[7 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 64;
        let dst_base = j * 32;
        let mut k = 0;
        while k < 32 {
            let x0 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + 32 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(), t);
            k += 4;
        }
    }
    // Pass 5: radix 16, groups 16
    for j in 0..16 {
        let w = twiddles[15 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 32;
        let dst_base = j * 16;
        let mut k = 0;
        while k < 16 {
            let x0 = _mm256_loadu_ps(data.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(data.as_ptr().add(src_base + 16 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(
                scratch.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(),
                t,
            );
            k += 4;
        }
    }
    // Pass 6: radix 32, groups 8
    for j in 0..32 {
        let w = twiddles[31 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 16;
        let dst_base = j * 8;
        let mut k = 0;
        while k < 8 {
            let x0 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(scratch.as_ptr().add(src_base + 8 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(data.as_mut_ptr().add(dst_base + 256 + k).cast::<f32>(), t);
            k += 4;
        }
    }
    // Passes 7-8: radix 64 then radix 128.
    stage_pair32_quarter_groups_two_avx_fma(
        data,
        scratch,
        64,
        &twiddles[63..127],
        &twiddles[127..255],
    );
    // Pass 9: radix 256, groups 1.
    stage32_groups_one_avx_fma(scratch, data, 256, &twiddles[255..511]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len8_32_avx_fma(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;
    // Pass 1: radix 1, groups 4
    for j in 0..1 {
        let w = twiddles[j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 8;
        let dst_base = j * 4;
        let mut k = 0;
        while k < 4 {
            let x0 = _mm256_loadu_ps(data.as_ptr().add(src_base + k).cast::<f32>());
            let x1 = _mm256_loadu_ps(data.as_ptr().add(src_base + 4 + k).cast::<f32>());
            let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
            let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
            let s = _mm256_add_ps(x0, product);
            let t = _mm256_sub_ps(x0, product);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + k).cast::<f32>(), s);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(dst_base + 4 + k).cast::<f32>(), t);
            k += 4;
        }
    }
    // Pass 2: radix 2, groups 2
    for j in 0..2 {
        let w = twiddles[1 + j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 4;
        let dst_base = j * 2;
        let src_vec = _mm256_loadu_ps(scratch.as_ptr().add(src_base).cast::<f32>());
        let x0 = _mm256_permute2f128_ps(src_vec, src_vec, 0x00);
        let x1 = _mm256_permute2f128_ps(src_vec, src_vec, 0x11);
        let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
        let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
        let s = _mm256_add_ps(x0, product);
        let t = _mm256_sub_ps(x0, product);
        _mm_storeu_ps(
            data.as_mut_ptr().add(dst_base).cast::<f32>(),
            _mm256_castps256_ps128(s),
        );
        _mm_storeu_ps(
            data.as_mut_ptr().add(dst_base + 4).cast::<f32>(),
            _mm256_castps256_ps128(t),
        );
    }
    // Pass 3: radix 4, groups 1
    for j in 0..4 {
        let w = twiddles[3 + j];
        let a = data[j * 2];
        let b = data[j * 2 + 1] * w;
        scratch[j] = a + b;
        scratch[j + 4] = a - b;
    }
    data.copy_from_slice(scratch);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn fixed_len4_32_avx_fma(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;
    // Pass 1: radix 1, groups 2
    for j in 0..1 {
        let w = twiddles[j];
        let w_re = _mm256_set1_ps(w.re);
        let w_im = _mm256_set1_ps(w.im);
        let src_base = j * 4;
        let dst_base = j * 2;
        let src_vec = _mm256_loadu_ps(data.as_ptr().add(src_base).cast::<f32>());
        let x0 = _mm256_permute2f128_ps(src_vec, src_vec, 0x00);
        let x1 = _mm256_permute2f128_ps(src_vec, src_vec, 0x11);
        let x1_shuf = _mm256_permute_ps(x1, 0b10110001);
        let product = _mm256_fmaddsub_ps(w_re, x1, _mm256_mul_ps(w_im, x1_shuf));
        let s = _mm256_add_ps(x0, product);
        let t = _mm256_sub_ps(x0, product);
        _mm_storeu_ps(
            scratch.as_mut_ptr().add(dst_base).cast::<f32>(),
            _mm256_castps256_ps128(s),
        );
        _mm_storeu_ps(
            scratch.as_mut_ptr().add(dst_base + 2).cast::<f32>(),
            _mm256_castps256_ps128(t),
        );
    }
    // Pass 2: radix 2, groups 1
    for j in 0..2 {
        let w = twiddles[1 + j];
        let a = scratch[j * 2];
        let b = scratch[j * 2 + 1] * w;
        data[j] = a + b;
        data[j + 2] = a - b;
    }
}

#[inline]
#[expect(clippy::too_many_arguments, reason = "register-blocked fused codelet")]
fn stage_triple_scalar_one_impl<C>(
    src: &[C],
    dst: &mut [C],
    src_base: usize,
    dst_base: usize,
    quarter_groups: usize,
    eighth_n: usize,
    quarter_n: usize,
    half_n: usize,
    k: usize,
    w1: C,
    w2a: C,
    w2b: C,
    w3a: C,
    w3b: C,
    w3c: C,
    w3d: C,
) where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    let x0 = src[src_base + k];
    let x1 = src[src_base + quarter_groups + k];
    let x2 = src[src_base + 2 * quarter_groups + k];
    let x3 = src[src_base + 3 * quarter_groups + k];
    let x4 = src[src_base + 4 * quarter_groups + k] * w1;
    let x5 = src[src_base + 5 * quarter_groups + k] * w1;
    let x6 = src[src_base + 6 * quarter_groups + k] * w1;
    let x7 = src[src_base + 7 * quarter_groups + k] * w1;

    let s0 = x0 + x4;
    let s1 = x1 + x5;
    let s2 = x2 + x6;
    let s3 = x3 + x7;
    let d0 = x0 - x4;
    let d1 = x1 - x5;
    let d2 = x2 - x6;
    let d3 = x3 - x7;

    let t2 = s2 * w2a;
    let t3 = s3 * w2a;
    let u2 = d2 * w2b;
    let u3 = d3 * w2b;
    let p0 = s0 + t2;
    let p1 = s1 + t3;
    let p4 = s0 - t2;
    let p5 = s1 - t3;
    let p2 = d0 + u2;
    let p3 = d1 + u3;
    let p6 = d0 - u2;
    let p7 = d1 - u3;

    let q0 = p1 * w3a;
    let q1 = p3 * w3b;
    let q2 = p5 * w3c;
    let q3 = p7 * w3d;
    let out_base = dst_base + k;
    dst[out_base] = p0 + q0;
    dst[half_n + out_base] = p0 - q0;
    dst[eighth_n + out_base] = p2 + q1;
    dst[half_n + eighth_n + out_base] = p2 - q1;
    dst[quarter_n + out_base] = p4 + q2;
    dst[half_n + quarter_n + out_base] = p4 - q2;
    dst[quarter_n + eighth_n + out_base] = p6 + q3;
    dst[half_n + quarter_n + eighth_n + out_base] = p6 - q3;
}

#[inline]
fn stage_triple_impl<C>(
    src: &[C],
    dst: &mut [C],
    radix: usize,
    first_twiddles: &[C],
    second_twiddles: &[C],
    third_twiddles: &[C],
) where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    let n = src.len();
    let groups = n / (radix << 1);
    let quarter_groups = groups >> 2;
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;

    for j in 0..radix {
        let src_base = j * groups * 2;
        let dst_base = j * quarter_groups;
        for k in 0..quarter_groups {
            stage_triple_scalar_one_impl(
                src,
                dst,
                src_base,
                dst_base,
                quarter_groups,
                eighth_n,
                quarter_n,
                half_n,
                k,
                first_twiddles[j],
                second_twiddles[j],
                second_twiddles[j + radix],
                third_twiddles[j],
                third_twiddles[j + radix],
                third_twiddles[j + 2 * radix],
                third_twiddles[j + 3 * radix],
            );
        }
    }
}

macro_rules! stockham_quad_unrolled {
    (
        $src:ident, $dst:ident, $src_base:expr, $dst_base:expr,
        $sixteenth_groups:expr, $sixteenth_n:expr, $k:expr,
        $w1:expr, $w20:expr, $w21:expr,
        $w30:expr, $w31:expr, $w32:expr, $w33:expr,
        $w40:expr, $w41:expr, $w42:expr, $w43:expr,
        $w44:expr, $w45:expr, $w46:expr, $w47:expr
    ) => {{
        let x0 = $src[$src_base + $k];
        let x1 = $src[$src_base + $sixteenth_groups + $k];
        let x2 = $src[$src_base + 2 * $sixteenth_groups + $k];
        let x3 = $src[$src_base + 3 * $sixteenth_groups + $k];
        let x4 = $src[$src_base + 4 * $sixteenth_groups + $k];
        let x5 = $src[$src_base + 5 * $sixteenth_groups + $k];
        let x6 = $src[$src_base + 6 * $sixteenth_groups + $k];
        let x7 = $src[$src_base + 7 * $sixteenth_groups + $k];
        let x8 = $src[$src_base + 8 * $sixteenth_groups + $k] * $w1;
        let x9 = $src[$src_base + 9 * $sixteenth_groups + $k] * $w1;
        let x10 = $src[$src_base + 10 * $sixteenth_groups + $k] * $w1;
        let x11 = $src[$src_base + 11 * $sixteenth_groups + $k] * $w1;
        let x12 = $src[$src_base + 12 * $sixteenth_groups + $k] * $w1;
        let x13 = $src[$src_base + 13 * $sixteenth_groups + $k] * $w1;
        let x14 = $src[$src_base + 14 * $sixteenth_groups + $k] * $w1;
        let x15 = $src[$src_base + 15 * $sixteenth_groups + $k] * $w1;

        let y0 = x0 + x8;
        let y1 = x1 + x9;
        let y2 = x2 + x10;
        let y3 = x3 + x11;
        let y4 = x4 + x12;
        let y5 = x5 + x13;
        let y6 = x6 + x14;
        let y7 = x7 + x15;
        let y8 = x0 - x8;
        let y9 = x1 - x9;
        let y10 = x2 - x10;
        let y11 = x3 - x11;
        let y12 = x4 - x12;
        let y13 = x5 - x13;
        let y14 = x6 - x14;
        let y15 = x7 - x15;

        let t4 = y4 * $w20;
        let t5 = y5 * $w20;
        let t6 = y6 * $w20;
        let t7 = y7 * $w20;
        let t12 = y12 * $w21;
        let t13 = y13 * $w21;
        let t14 = y14 * $w21;
        let t15 = y15 * $w21;

        let z0 = y0 + t4;
        let z1 = y1 + t5;
        let z2 = y2 + t6;
        let z3 = y3 + t7;
        let z4 = y8 + t12;
        let z5 = y9 + t13;
        let z6 = y10 + t14;
        let z7 = y11 + t15;
        let z8 = y0 - t4;
        let z9 = y1 - t5;
        let z10 = y2 - t6;
        let z11 = y3 - t7;
        let z12 = y8 - t12;
        let z13 = y9 - t13;
        let z14 = y10 - t14;
        let z15 = y11 - t15;

        let u2 = z2 * $w30;
        let u3 = z3 * $w30;
        let u6 = z6 * $w31;
        let u7 = z7 * $w31;
        let u10 = z10 * $w32;
        let u11 = z11 * $w32;
        let u14 = z14 * $w33;
        let u15 = z15 * $w33;

        let p0 = z0 + u2;
        let p1 = z1 + u3;
        let p2 = z4 + u6;
        let p3 = z5 + u7;
        let p4 = z8 + u10;
        let p5 = z9 + u11;
        let p6 = z12 + u14;
        let p7 = z13 + u15;
        let p8 = z0 - u2;
        let p9 = z1 - u3;
        let p10 = z4 - u6;
        let p11 = z5 - u7;
        let p12 = z8 - u10;
        let p13 = z9 - u11;
        let p14 = z12 - u14;
        let p15 = z13 - u15;

        let q1 = p1 * $w40;
        let q3 = p3 * $w41;
        let q5 = p5 * $w42;
        let q7 = p7 * $w43;
        let q9 = p9 * $w44;
        let q11 = p11 * $w45;
        let q13 = p13 * $w46;
        let q15 = p15 * $w47;

        let out_base = $dst_base + $k;
        $dst[out_base] = p0 + q1;
        $dst[out_base + $sixteenth_n] = p2 + q3;
        $dst[out_base + 2 * $sixteenth_n] = p4 + q5;
        $dst[out_base + 3 * $sixteenth_n] = p6 + q7;
        $dst[out_base + 4 * $sixteenth_n] = p8 + q9;
        $dst[out_base + 5 * $sixteenth_n] = p10 + q11;
        $dst[out_base + 6 * $sixteenth_n] = p12 + q13;
        $dst[out_base + 7 * $sixteenth_n] = p14 + q15;
        $dst[out_base + 8 * $sixteenth_n] = p0 - q1;
        $dst[out_base + 9 * $sixteenth_n] = p2 - q3;
        $dst[out_base + 10 * $sixteenth_n] = p4 - q5;
        $dst[out_base + 11 * $sixteenth_n] = p6 - q7;
        $dst[out_base + 12 * $sixteenth_n] = p8 - q9;
        $dst[out_base + 13 * $sixteenth_n] = p10 - q11;
        $dst[out_base + 14 * $sixteenth_n] = p12 - q13;
        $dst[out_base + 15 * $sixteenth_n] = p14 - q15;
    }};
}

/// Fuses four adjacent radix-2 Stockham stages as a radix-16 autosort codelet.
///
/// ## Proof sketch
///
/// Let `r` be the incoming Stockham stride and `G = N/(2r)`. A four-stage
/// fusion is valid only when `G >= 8`; each independent work item is identified
/// by `(j, k)` with `j < r` and `k < G/8`. The 16 inputs are
/// `x_m = src[j*2G + m*(G/8) + k]`.
///
/// For local stage `t` there are `2^t` branches. Branch `b` applies the exact
/// scalar Stockham recurrence to the two halves of a local block of length
/// `16/2^t`, using twiddle table element `W_t[j + b*r]`, then writes sums to
/// the low local half and differences to the high local half. This is the same
/// index relation as `stage64`, restricted to the 16 values reachable from
/// `(j, k)`. Induction over the four local stages proves that the final local
/// array equals four scalar Stockham passes, and storing local index `m` at
/// `dst[j*(G/8) + m*(N/16) + k]` preserves the global autosort order. Only
/// fixed-size stack arrays are used, so the codelet has no heap scratch.
#[inline]
fn stage_quad_impl<C>(
    src: &[C],
    dst: &mut [C],
    radix: usize,
    first_twiddles: &[C],
    second_twiddles: &[C],
    third_twiddles: &[C],
    fourth_twiddles: &[C],
) where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    let n = src.len();
    let groups = n / (radix << 1);
    let sixteenth_groups = groups >> 3;
    let sixteenth_n = n >> 4;

    for j in 0..radix {
        let src_base = j * groups * 2;
        let dst_base = j * sixteenth_groups;
        let w1 = first_twiddles[j];
        let w20 = second_twiddles[j];
        let w21 = second_twiddles[j + radix];
        let w30 = third_twiddles[j];
        let w31 = third_twiddles[j + radix];
        let w32 = third_twiddles[j + 2 * radix];
        let w33 = third_twiddles[j + 3 * radix];
        let w40 = fourth_twiddles[j];
        let w41 = fourth_twiddles[j + radix];
        let w42 = fourth_twiddles[j + 2 * radix];
        let w43 = fourth_twiddles[j + 3 * radix];
        let w44 = fourth_twiddles[j + 4 * radix];
        let w45 = fourth_twiddles[j + 5 * radix];
        let w46 = fourth_twiddles[j + 6 * radix];
        let w47 = fourth_twiddles[j + 7 * radix];
        for k in 0..sixteenth_groups {
            stockham_quad_unrolled!(
                src,
                dst,
                src_base,
                dst_base,
                sixteenth_groups,
                sixteenth_n,
                k,
                w1,
                w20,
                w21,
                w30,
                w31,
                w32,
                w33,
                w40,
                w41,
                w42,
                w43,
                w44,
                w45,
                w46,
                w47
            );
        }
    }
}

/// Fuses two adjacent radix-2 Stockham stages.
///
/// For a starting stage radix `r`, the scalar Stockham recurrence first forms
/// `a = x_0 + w_j x_2` and `b = x_0 - w_j x_2` inside each length-`2r`
/// group. The next stage applies the same recurrence at radix `2r` to the
/// adjacent `k` pairs. Substituting the first equations into the second gives
/// the four outputs below with twiddles `{w_j, w_j', w_{j+r}'}`. The function
/// evaluates that substitution directly, so it removes one full scratch
/// traversal while producing byte-for-byte the same stage order as two scalar
/// Stockham passes.
#[inline]
fn stage_pair_impl<C>(
    src: &[C],
    dst: &mut [C],
    radix: usize,
    first_twiddles: &[C],
    second_twiddles: &[C],
) where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    if radix == 1 {
        let n = src.len();
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let w3 = second_twiddles[1];
        for k in 0..quarter_n {
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
        }
        return;
    }

    let n = src.len();
    let groups = n / (radix << 1);
    let half_groups = groups >> 1;
    let quarter_n = n >> 2;
    let half_n = n >> 1;

    for j in 0..radix {
        let w1 = first_twiddles[j];
        let w2 = second_twiddles[j];
        let w3 = second_twiddles[j + radix];
        let src_base = j * groups * 2;
        let dst_base = j * half_groups;
        for k in 0..half_groups {
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
        }
    }
}

unsafe fn forward32_avx_with_scratch(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    if data.len() == 4 {
        fixed_len4_32_avx_fma(data, scratch, twiddles);
        return;
    }
    if data.len() == 8 {
        fixed_len8_32_avx_fma(data, scratch, twiddles);
        return;
    }
    if data.len() == 64 {
        fixed_len64_32_avx_fma(data, scratch, twiddles);
        return;
    }
    if data.len() == 4096 {
        transform_len4096_four_triples::<F32StockhamAvxFma>(data, scratch, twiddles);
        return;
    }
    transform::<F32StockhamAvxFma>(data, scratch, twiddles, None);
}

pub(crate) trait StockhamKernel: Sized {
    type Complex;

    /// Forward radix-2 Stockham FFT into natural order using caller-provided scratch.
    ///
    /// `data` and `scratch` must have the same length (a power of two).
    /// `twiddles` must be the output of the matching `build_forward_twiddle_table_*` call.
    fn forward_with_scratch(
        data: &mut [Self::Complex],
        scratch: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    );
}

impl StockhamKernel for f64 {
    type Complex = Complex64;

    #[inline]
    fn forward_with_scratch(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
    ) {
        let n = data.len();
        debug_assert_eq!(scratch.len(), n, "stockham scratch length mismatch");
        debug_assert!(n.is_power_of_two());
        if n <= 1 {
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        {
            unsafe { forward64_avx_with_scratch(data, scratch, twiddles) };
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
        {
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("fma")
            {
                unsafe { forward64_avx_with_scratch(data, scratch, twiddles) };
                return;
            }
            transform::<F64Stockham>(data, scratch, twiddles, None);
        }
    }
}

impl StockhamKernel for f32 {
    type Complex = Complex32;

    #[inline]
    fn forward_with_scratch(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
    ) {
        let n = data.len();
        debug_assert_eq!(scratch.len(), n, "stockham scratch length mismatch");
        debug_assert!(n.is_power_of_two());
        if n <= 1 {
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        {
            unsafe { forward32_avx_with_scratch(data, scratch, twiddles) };
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
        {
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("fma")
            {
                unsafe { forward32_avx_with_scratch(data, scratch, twiddles) };
                return;
            }
            transform::<F32Stockham>(data, scratch, twiddles, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f32_avx_groups_eight_quad_stage_matches_scalar_reference() {
        if !std::arch::is_x86_feature_detected!("avx")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let radix = 64usize;
        let n = radix << 4;
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.013).sin(), (k as f32 * 0.019).cos()))
            .collect();
        let twiddles =
            crate::application::execution::kernel::radix2::build_forward_twiddle_table_32(n);
        let base = radix - 1;
        let first = &twiddles[base..base + radix];
        let second = &twiddles[base + radix..base + 3 * radix];
        let third = &twiddles[base + 3 * radix..base + 7 * radix];
        let fourth = &twiddles[base + 7 * radix..base + 15 * radix];
        let mut expected = vec![Complex32::new(0.0, 0.0); n];
        let mut actual = expected.clone();

        stage_quad_impl(&input, &mut expected, radix, first, second, third, fourth);
        <F32StockhamAvxFma as StockhamPrecision>::stage_quad(
            &input,
            &mut actual,
            radix,
            first,
            second,
            third,
            fourth,
        );

        let err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f32::max);
        assert!(
            err < 1.0e-4,
            "groups-eight f32 AVX quad stage err={err:.2e}"
        );
    }

    #[test]
    fn scalar_fallback_policy_types_remain_reachable_in_tests() {
        let _ = F64Stockham;
        let _ = F32Stockham;
        assert_eq!(<F64Stockham as StockhamPrecision>::MAX_FUSED_STAGES, 4);
        assert_eq!(<F32Stockham as StockhamPrecision>::MAX_FUSED_STAGES, 4);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn stockham_scheduler_uses_copyback_instead_of_stride1_prepass() {
        let source = include_str!("stockham.rs");
        let body = source
            .split_once("fn transform<P: StockhamPrecision>(")
            .and_then(|(_, tail)| tail.split_once("#[cfg(target_arch = \"x86_64\")]"))
            .map(|(body, _)| body)
            .expect("generic Stockham transform body must be present");
        assert!(!body.contains("schedule_odd_flips::<P>"));
        assert!(!body.contains("prepass_twiddles"));
        assert!(body.contains("data.copy_from_slice(scratch);"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn butterfly512_f32_packed_twiddles_match_separated_column_contract() {
        let twiddles =
            crate::application::execution::kernel::radix2::build_forward_twiddle_table_32(512);
        let packed = build_butterfly512_twiddles_32(&twiddles);

        assert_eq!(packed.len(), 120);
        for columnset in 0..8 {
            let col_base = columnset * 4;
            for row in 1..16 {
                let vector = packed[columnset * 15 + row - 1];
                for lane in 0..4 {
                    let expected =
                        stockham_mixed_twiddle_32::<16, 32>(&twiddles, row, col_base + lane);
                    assert_eq!(
                        vector[lane],
                        expected,
                        "f32 packed twiddle row={row} col={}",
                        col_base + lane
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn butterfly512_f64_packed_twiddles_match_separated_column_contract() {
        let twiddles =
            crate::application::execution::kernel::radix2::build_forward_twiddle_table_64(512);
        let packed = build_butterfly512_twiddles_64(&twiddles);

        assert_eq!(packed.len(), 240);
        for columnset in 0..16 {
            let col_base = columnset * 2;
            for row in 1..16 {
                let vector = packed[columnset * 15 + row - 1];
                for lane in 0..2 {
                    let expected =
                        stockham_mixed_twiddle_64::<16, 32>(&twiddles, row, col_base + lane);
                    assert_eq!(
                        vector[lane],
                        expected,
                        "f64 packed twiddle row={row} col={}",
                        col_base + lane
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_scheduler_selects_f32_n4096_tail_shape() {
        assert!(<F64StockhamAvxFma as StockhamPrecision>::stage_triple_enabled(64, 4096, true));
        assert!(!<F64StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(256, 4096, false));
        assert!(!<F64StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(64, 1024, true));

        assert!(<F32StockhamAvxFma as StockhamPrecision>::stage_triple_enabled(64, 4096, true));
        assert!(!<F32StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(256, 4096, false));
        assert!(!<F32StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(64, 1024, true));
        assert!(<F32StockhamAvxFma as StockhamPrecision>::stage_triple_enabled(512, 8192, false));
        assert!(!<F32StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(512, 8192, false));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f64_triple_avx_routes_groups_eight_to_dedicated_late_leaf() {
        let source = include_str!("stockham.rs");
        let body = source
            .split_once("impl StockhamPrecision for F64StockhamAvxFma")
            .and_then(|(_, tail)| tail.split_once("impl StockhamPrecision for F32StockhamAvxFma"))
            .map(|(body, _)| body)
            .expect("F64StockhamAvxFma implementation must be present");

        assert!(body.contains("groups == 8"));
        assert!(body.contains("stage_triple64_groups_eight_avx_fma("));
        assert!(!body.contains("bit_reverse"));
        assert!(!body.contains("reverse_bits"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f64_avx_groups_eight_triple_stage_matches_scalar_reference() {
        if !std::arch::is_x86_feature_detected!("avx")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let radix = 64usize;
        let n = radix << 4;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.013).sin(), (k as f64 * 0.019).cos()))
            .collect();
        let twiddles =
            crate::application::execution::kernel::radix2::build_forward_twiddle_table_64(n);
        let base = radix - 1;
        let first = &twiddles[base..base + radix];
        let second = &twiddles[base + radix..base + 3 * radix];
        let third = &twiddles[base + 3 * radix..base + 7 * radix];
        let mut expected = vec![Complex64::new(0.0, 0.0); n];
        let mut actual = expected.clone();

        stage_triple_impl(&input, &mut expected, radix, first, second, third);
        unsafe {
            stage_triple64_groups_eight_avx_fma(&input, &mut actual, radix, first, second, third)
        };

        let err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(
            err < 1.0e-12,
            "groups-eight f64 AVX triple stage err={err:.2e}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f64_hybrid_radix8x512_matches_stockham_n4096() {
        if !std::arch::is_x86_feature_detected!("avx")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let n = 4096usize;
        let mut expected: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.007).sin(), (k as f64 * 0.011).cos()))
            .collect();
        let mut actual = expected.clone();
        let twiddles =
            crate::application::execution::kernel::radix2::build_forward_twiddle_table_64(n);
        let mut expected_scratch = vec![Complex64::new(0.0, 0.0); n];
        let mut actual_scratch = vec![Complex64::new(0.0, 0.0); n];

        transform::<F64StockhamAvxFma>(&mut expected, &mut expected_scratch, &twiddles, None);
        unsafe {
            hybrid_radix8x512_64_avx_fma::<false>(&mut actual, &mut actual_scratch, &twiddles);
        }

        let err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(err < 1.0e-10, "f64 hybrid radix8x512 err={err:.2e}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f32_hybrid_radix8x512_matches_stockham_n4096() {
        if !std::arch::is_x86_feature_detected!("avx")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let n = 4096usize;
        let mut expected: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.007).sin(), (k as f32 * 0.011).cos()))
            .collect();
        let mut actual = expected.clone();
        let twiddles =
            crate::application::execution::kernel::radix2::build_forward_twiddle_table_32(n);
        let mut expected_scratch = vec![Complex32::new(0.0, 0.0); n];
        let mut actual_scratch = vec![Complex32::new(0.0, 0.0); n];

        transform::<F32StockhamAvxFma>(&mut expected, &mut expected_scratch, &twiddles, None);
        unsafe {
            hybrid_radix8x512_32_avx_fma::<false>(&mut actual, &mut actual_scratch, &twiddles);
        }

        let err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f32::max);
        let tolerance = (n as f32 / 2.0) * f32::EPSILON;
        assert!(
            err < tolerance,
            "f32 hybrid radix8x512 err={err:.2e} tolerance={tolerance:.2e}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f64_hybrid_radix8x512_inverse_roundtrip_n4096() {
        if !std::arch::is_x86_feature_detected!("avx")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let n = 4096usize;
        let mut data: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.017).sin(), (k as f64 * 0.023).cos()))
            .collect();
        let original = data.clone();
        let forward =
            crate::application::execution::kernel::radix2::build_forward_twiddle_table_64(n);
        let inverse =
            crate::application::execution::kernel::radix2::build_inverse_twiddle_table_64(n);
        let mut scratch = vec![Complex64::new(0.0, 0.0); n];

        unsafe {
            hybrid_radix8x512_64_avx_fma::<false>(&mut data, &mut scratch, &forward);
            hybrid_radix8x512_64_avx_fma::<true>(&mut data, &mut scratch, &inverse);
        }
        data.iter_mut().for_each(|value| *value *= 1.0 / n as f64);

        let err = data
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(err < 1.0e-10, "f64 hybrid inverse roundtrip err={err:.2e}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f32_hybrid_radix8x512_inverse_roundtrip_n4096() {
        if !std::arch::is_x86_feature_detected!("avx")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let n = 4096usize;
        let mut data: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.017).sin(), (k as f32 * 0.023).cos()))
            .collect();
        let original = data.clone();
        let forward =
            crate::application::execution::kernel::radix2::build_forward_twiddle_table_32(n);
        let inverse =
            crate::application::execution::kernel::radix2::build_inverse_twiddle_table_32(n);
        let mut scratch = vec![Complex32::new(0.0, 0.0); n];

        unsafe {
            hybrid_radix8x512_32_avx_fma::<false>(&mut data, &mut scratch, &forward);
            hybrid_radix8x512_32_avx_fma::<true>(&mut data, &mut scratch, &inverse);
        }
        data.iter_mut().for_each(|value| *value *= 1.0 / n as f32);

        let err = data
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f32::max);
        assert!(err < 1.0e-4, "f32 hybrid inverse roundtrip err={err:.2e}");
    }

    #[test]
    fn hybrid_radix8x512_source_has_no_bit_reversal_or_allocation() {
        let source = include_str!("stockham.rs");
        let body = source
            .split_once("unsafe fn hybrid_radix8x512_64_avx_fma")
            .and_then(|(_, tail)| tail.split_once("unsafe fn forward64_avx_with_scratch"))
            .map(|(body, _)| body)
            .expect("hybrid radix8x512 body must be present");

        assert!(!body.contains("bit_reverse"));
        assert!(!body.contains("reverse_bits"));
        assert!(!body.contains("bitrev"));
        assert!(!body.contains("Vec<"));
        assert!(!body.contains("vec!"));
        assert!(!body.contains("Box<"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f64_avx_schedule_roundtrip_holds_for_n8192() {
        if !std::arch::is_x86_feature_detected!("avx")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let mut data: Vec<Complex64> = (0..8192)
            .map(|k| Complex64::new((k as f64 * 0.007).sin(), (k as f64 * 0.011).cos()))
            .collect();
        let original = data.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); data.len()];
        let forward = crate::application::execution::kernel::radix2::build_forward_twiddle_table_64(
            data.len(),
        );
        let inverse = crate::application::execution::kernel::radix2::build_inverse_twiddle_table_64(
            data.len(),
        );

        f64::forward_with_scratch(&mut data, &mut scratch, &forward);
        // inverse_with_scratch removed: implement as forward on inverse twiddles + 1/N scale.
        f64::forward_with_scratch(&mut data, &mut scratch, &inverse);
        let scale = 1.0 / data.len() as f64;
        for v in &mut data {
            *v *= scale;
        }

        let err = data
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(
            err < 1.0e-10,
            "n8192 f64 AVX Stockham roundtrip err={err:.2e}"
        );
    }
}
