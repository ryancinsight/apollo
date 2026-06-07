//! AVX2+FMA flat radix-4 Stockham pass for f64 and f32.
//!
//! Includes DFT-4 butterflies for __m256d (f64) and __m256/__m128 (f32).

use super::helpers::{
    apply_pointwise_f32, apply_pointwise_f64, cmul, cmul_f32, cmul_f32_128, rot_neg_i,
    rot_neg_i_f32, rot_pos_i, rot_pos_i_f32,
};
use num_complex::Complex;
use std::arch::x86_64::{
    __m128, __m256, __m256d, _mm256_add_pd, _mm256_add_ps, _mm256_castpd256_pd128,
    _mm256_castps256_ps128, _mm256_extractf128_pd, _mm256_extractf128_ps, _mm256_loadu_pd,
    _mm256_loadu_ps, _mm256_set_m128, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_sub_pd,
    _mm256_sub_ps, _mm_add_ps, _mm_castpd_ps, _mm_castps_pd, _mm_loadu_ps, _mm_permute_ps,
    _mm_prefetch, _mm_set_ps, _mm_storeu_pd, _mm_storeu_ps, _mm_sub_ps, _mm_unpackhi_pd,
    _mm_unpacklo_pd, _mm_xor_ps, _MM_HINT_T0,
};

// ── f64 DFT-4 butterfly ─────────────────────────────────────────────────────

/// Radix-4 butterfly for 2 complex-f64 pairs simultaneously.
/// Returns (b0, b1, b2, b3) given arms (a0, a1, a2, a3).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft4_f64<const INVERSE: bool>(
    a0: __m256d,
    a1: __m256d,
    a2: __m256d,
    a3: __m256d,
) -> (__m256d, __m256d, __m256d, __m256d) {
    let t0 = _mm256_add_pd(a0, a2);
    let t1 = _mm256_sub_pd(a0, a2);
    let t2 = _mm256_add_pd(a1, a3);
    let t3 = _mm256_sub_pd(a1, a3);
    if INVERSE {
        let r = rot_pos_i(t3); // +i·t3
        (
            _mm256_add_pd(t0, t2),
            _mm256_add_pd(t1, r),
            _mm256_sub_pd(t0, t2),
            _mm256_sub_pd(t1, r),
        )
    } else {
        let r = rot_neg_i(t3); // -i·t3
        (
            _mm256_add_pd(t0, t2),
            _mm256_add_pd(t1, r),
            _mm256_sub_pd(t0, t2),
            _mm256_sub_pd(t1, r),
        )
    }
}

// ── f64 radix-4 pass ────────────────────────────────────────────────────────

/// Flat radix-4 Stockham pass in AVX2+FMA for f64. Processes all `g_count` groups.
///
/// `src` and `dst` are the full n-element arrays for this pass (length n = g_count × stage_chunk).
/// `tw` is the twiddle slice for this stage: `(R-1) × prev_len` entries, arm k at `[(k-1)*prev_len..]`.
///
/// # Layout invariants
/// - stage_chunk = 4 × prev_len
/// - stride = g_count × prev_len = n / 4  (constant for r=4 regardless of prev_len)
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r4_f64<
    const INVERSE: bool,
>(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    prev_len: usize,
    g_count: usize,
    stage_chunk: usize,
    tw: &[Complex<f64>],
    pointwise: Option<&[Complex<f64>]>,
) {
    let stride = g_count * prev_len; // = n/4 for r=4
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        // No twiddles (j=0 always, W^0=1). Process 2 groups per AVX2 iteration.
        let mut g = 0usize;
        while g + 1 < g_count {
            let a0 = _mm256_loadu_pd(src_ptr.add(g).cast::<f64>());
            let a1 = _mm256_loadu_pd(src_ptr.add(stride + g).cast::<f64>());
            let a2 = _mm256_loadu_pd(src_ptr.add(2 * stride + g).cast::<f64>());
            let a3 = _mm256_loadu_pd(src_ptr.add(3 * stride + g).cast::<f64>());

            let (b0, b1, b2, b3) = dft4_f64::<INVERSE>(a0, a1, a2, a3);

            let d0 = dst_ptr.add(g * 4);
            _mm_storeu_pd(d0.add(0).cast::<f64>(), _mm256_castpd256_pd128(b0));
            _mm_storeu_pd(d0.add(1).cast::<f64>(), _mm256_castpd256_pd128(b1));
            _mm_storeu_pd(d0.add(2).cast::<f64>(), _mm256_castpd256_pd128(b2));
            _mm_storeu_pd(d0.add(3).cast::<f64>(), _mm256_castpd256_pd128(b3));
            let d1 = dst_ptr.add((g + 1) * 4);
            _mm_storeu_pd(d1.add(0).cast::<f64>(), _mm256_extractf128_pd(b0, 1));
            _mm_storeu_pd(d1.add(1).cast::<f64>(), _mm256_extractf128_pd(b1, 1));
            _mm_storeu_pd(d1.add(2).cast::<f64>(), _mm256_extractf128_pd(b2, 1));
            _mm_storeu_pd(d1.add(3).cast::<f64>(), _mm256_extractf128_pd(b3, 1));

            g += 2;
        }
        // Scalar tail (odd g_count).
        if g < g_count {
            let src_base = g;
            let d = dst_ptr.add(g * 4);
            let a0s = *src_ptr.add(src_base);
            let a1s = *src_ptr.add(stride + src_base);
            let a2s = *src_ptr.add(2 * stride + src_base);
            let a3s = *src_ptr.add(3 * stride + src_base);
            let t0 = a0s + a2s;
            let t1 = a0s - a2s;
            let t2 = a1s + a3s;
            let t3 = a1s - a3s;
            let (it3_re, it3_im) = if INVERSE {
                (-t3.im, t3.re)
            } else {
                (t3.im, -t3.re)
            };
            let it3 = Complex {
                re: it3_re,
                im: it3_im,
            };
            *d.add(0) = t0 + t2;
            *d.add(1) = t1 + it3;
            *d.add(2) = t0 - t2;
            *d.add(3) = t1 - it3;
        }
    } else {
        // prev_len >= 2: twiddles applied.
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;

            let mut j = 0usize;

            // Width-4 loop
            while j + 3 < prev_len {
                let off0 = src_base + j;
                let off1 = stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;
                let a0_lo = _mm256_loadu_pd(src_ptr.add(off0).cast::<f64>());
                let a1_lo_raw = _mm256_loadu_pd(src_ptr.add(off1).cast::<f64>());
                let a2_lo_raw = _mm256_loadu_pd(src_ptr.add(off2).cast::<f64>());
                let a3_lo_raw = _mm256_loadu_pd(src_ptr.add(off3).cast::<f64>());
                let a0_hi = _mm256_loadu_pd(src_ptr.add(off0 + 2).cast::<f64>());
                let a1_hi_raw = _mm256_loadu_pd(src_ptr.add(off1 + 2).cast::<f64>());
                let a2_hi_raw = _mm256_loadu_pd(src_ptr.add(off2 + 2).cast::<f64>());
                let a3_hi_raw = _mm256_loadu_pd(src_ptr.add(off3 + 2).cast::<f64>());
                let tw1_lo = _mm256_loadu_pd(tw_ptr.add(j).cast::<f64>());
                let tw2_lo = _mm256_loadu_pd(tw_ptr.add(prev_len + j).cast::<f64>());
                let tw3_lo = _mm256_loadu_pd(tw_ptr.add(2 * prev_len + j).cast::<f64>());
                let tw1_hi = _mm256_loadu_pd(tw_ptr.add(j + 2).cast::<f64>());
                let tw2_hi = _mm256_loadu_pd(tw_ptr.add(prev_len + j + 2).cast::<f64>());
                let tw3_hi = _mm256_loadu_pd(tw_ptr.add(2 * prev_len + j + 2).cast::<f64>());
                if prev_len >= 16 {
                    _mm_prefetch(tw_ptr.add(j + 8).cast::<i8>(), _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(prev_len + j + 8).cast::<i8>(), _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(2 * prev_len + j + 8).cast::<i8>(), _MM_HINT_T0);
                }
                let a1_lo = cmul(a1_lo_raw, tw1_lo);
                let a2_lo = cmul(a2_lo_raw, tw2_lo);
                let a3_lo = cmul(a3_lo_raw, tw3_lo);
                let a1_hi = cmul(a1_hi_raw, tw1_hi);
                let a2_hi = cmul(a2_hi_raw, tw2_hi);
                let a3_hi = cmul(a3_hi_raw, tw3_hi);
                let (b0_lo, b1_lo, b2_lo, b3_lo) = dft4_f64::<INVERSE>(a0_lo, a1_lo, a2_lo, a3_lo);
                let (b0_hi, b1_hi, b2_hi, b3_hi) = dft4_f64::<INVERSE>(a0_hi, a1_hi, a2_hi, a3_hi);
                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_pd(dp.add(j).cast::<f64>(), b0_lo);
                _mm256_storeu_pd(dp.add(j + 2).cast::<f64>(), b0_hi);
                _mm256_storeu_pd(dp.add(j + prev_len).cast::<f64>(), b1_lo);
                _mm256_storeu_pd(dp.add(j + 2 + prev_len).cast::<f64>(), b1_hi);
                _mm256_storeu_pd(dp.add(j + 2 * prev_len).cast::<f64>(), b2_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 2 * prev_len).cast::<f64>(), b2_hi);
                _mm256_storeu_pd(dp.add(j + 3 * prev_len).cast::<f64>(), b3_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 3 * prev_len).cast::<f64>(), b3_hi);
                j += 4;
            }

            // Width-2 loop
            while j + 1 < prev_len {
                let off0 = src_base + j;
                let off1 = stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;

                let a0 = _mm256_loadu_pd(src_ptr.add(off0).cast::<f64>());
                let mut a1 = _mm256_loadu_pd(src_ptr.add(off1).cast::<f64>());
                let mut a2 = _mm256_loadu_pd(src_ptr.add(off2).cast::<f64>());
                let mut a3 = _mm256_loadu_pd(src_ptr.add(off3).cast::<f64>());

                let tw1 = _mm256_loadu_pd(tw_ptr.add(j).cast::<f64>());
                let tw2 = _mm256_loadu_pd(tw_ptr.add(prev_len + j).cast::<f64>());
                let tw3 = _mm256_loadu_pd(tw_ptr.add(2 * prev_len + j).cast::<f64>());
                a1 = cmul(a1, tw1);
                a2 = cmul(a2, tw2);
                a3 = cmul(a3, tw3);

                let (b0, b1, b2, b3) = dft4_f64::<INVERSE>(a0, a1, a2, a3);

                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_pd(dp.add(j).cast::<f64>(), b0);
                _mm256_storeu_pd(dp.add(j + prev_len).cast::<f64>(), b1);
                _mm256_storeu_pd(dp.add(j + 2 * prev_len).cast::<f64>(), b2);
                _mm256_storeu_pd(dp.add(j + 3 * prev_len).cast::<f64>(), b3);

                j += 2;
            }

            // Scalar tail
            if j < prev_len {
                let src_base_j = src_base + j;
                let a0s = *src_ptr.add(src_base_j);
                let mut a1s = *src_ptr.add(stride + src_base_j);
                let mut a2s = *src_ptr.add(2 * stride + src_base_j);
                let mut a3s = *src_ptr.add(3 * stride + src_base_j);
                if j > 0 {
                    let tw1 = *tw.as_ptr().add(j);
                    let tw2 = *tw.as_ptr().add(prev_len + j);
                    let tw3 = *tw.as_ptr().add(2 * prev_len + j);
                    a1s = Complex {
                        re: a1s.re * tw1.re - a1s.im * tw1.im,
                        im: a1s.re * tw1.im + a1s.im * tw1.re,
                    };
                    a2s = Complex {
                        re: a2s.re * tw2.re - a2s.im * tw2.im,
                        im: a2s.re * tw2.im + a2s.im * tw2.re,
                    };
                    a3s = Complex {
                        re: a3s.re * tw3.re - a3s.im * tw3.im,
                        im: a3s.re * tw3.im + a3s.im * tw3.re,
                    };
                }
                let t0 = a0s + a2s;
                let t1 = a0s - a2s;
                let t2 = a1s + a3s;
                let t3 = a1s - a3s;
                let (it3_re, it3_im) = if INVERSE {
                    (-t3.im, t3.re)
                } else {
                    (t3.im, -t3.re)
                };
                let it3 = Complex {
                    re: it3_re,
                    im: it3_im,
                };
                let dp = dst_ptr.add(dst_base);
                *dp.add(j) = t0 + t2;
                *dp.add(j + prev_len) = t1 + it3;
                *dp.add(j + 2 * prev_len) = t0 - t2;
                *dp.add(j + 3 * prev_len) = t1 - it3;
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f64(dst_ptr, pw.as_ptr(), dst.len());
    }
}

// ── f32 DFT-4 butterflies ────────────────────────────────────────────────────

/// Radix-4 butterfly for 4 Complex<f32> simultaneously.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft4_f32<const INVERSE: bool>(
    a0: __m256,
    a1: __m256,
    a2: __m256,
    a3: __m256,
) -> (__m256, __m256, __m256, __m256) {
    let t0 = _mm256_add_ps(a0, a2);
    let t1 = _mm256_sub_ps(a0, a2);
    let t2 = _mm256_add_ps(a1, a3);
    let t3 = _mm256_sub_ps(a1, a3);
    if INVERSE {
        let r = rot_pos_i_f32(t3);
        (
            _mm256_add_ps(t0, t2),
            _mm256_add_ps(t1, r),
            _mm256_sub_ps(t0, t2),
            _mm256_sub_ps(t1, r),
        )
    } else {
        let r = rot_neg_i_f32(t3);
        (
            _mm256_add_ps(t0, t2),
            _mm256_add_ps(t1, r),
            _mm256_sub_ps(t0, t2),
            _mm256_sub_ps(t1, r),
        )
    }
}

/// Radix-4 butterfly for 2 Complex<f32> simultaneously (128-bit lane pair).
/// Used when prev_len is 2 or 3 (less than 4) to process 2 columns at a time.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft4_f32_128<const INVERSE: bool>(
    a0: __m128,
    a1: __m128,
    a2: __m128,
    a3: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let t0 = _mm_add_ps(a0, a2);
    let t1 = _mm_sub_ps(a0, a2);
    let t2 = _mm_add_ps(a1, a3);
    let t3 = _mm_sub_ps(a1, a3);
    if INVERSE {
        let perm = _mm_permute_ps(t3, 0xB1);
        let sign = _mm_set_ps(0.0, -0.0, 0.0, -0.0);
        let r = _mm_xor_ps(perm, sign);
        (
            _mm_add_ps(t0, t2),
            _mm_add_ps(t1, r),
            _mm_sub_ps(t0, t2),
            _mm_sub_ps(t1, r),
        )
    } else {
        let perm = _mm_permute_ps(t3, 0xB1);
        let sign = _mm_set_ps(-0.0, 0.0, -0.0, 0.0);
        let r = _mm_xor_ps(perm, sign);
        (
            _mm_add_ps(t0, t2),
            _mm_add_ps(t1, r),
            _mm_sub_ps(t0, t2),
            _mm_sub_ps(t1, r),
        )
    }
}

// ── f32 radix-4 pass ────────────────────────────────────────────────────────

/// Flat radix-4 Stockham pass in AVX2+FMA for f32. Processes all `g_count` groups.
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r4_f32<
    const INVERSE: bool,
>(
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    prev_len: usize,
    g_count: usize,
    stage_chunk: usize,
    tw: &[Complex<f32>],
    pointwise: Option<&[Complex<f32>]>,
) {
    let stride = g_count * prev_len; // = n/4 for r=4
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        // No twiddles. Process 4 groups per AVX2 iteration.
        let mut g = 0usize;
        while g + 3 < g_count {
            let a0 = _mm256_loadu_ps(src_ptr.add(g).cast::<f32>());
            let a1 = _mm256_loadu_ps(src_ptr.add(stride + g).cast::<f32>());
            let a2 = _mm256_loadu_ps(src_ptr.add(2 * stride + g).cast::<f32>());
            let a3 = _mm256_loadu_ps(src_ptr.add(3 * stride + g).cast::<f32>());

            let (b0, b1, b2, b3) = dft4_f32::<INVERSE>(a0, a1, a2, a3);

            let b0_lo = _mm_castps_pd(_mm256_castps256_ps128(b0));
            let b1_lo = _mm_castps_pd(_mm256_castps256_ps128(b1));
            let b2_lo = _mm_castps_pd(_mm256_castps256_ps128(b2));
            let b3_lo = _mm_castps_pd(_mm256_castps256_ps128(b3));
            let b0_hi = _mm_castps_pd(_mm256_extractf128_ps(b0, 1));
            let b1_hi = _mm_castps_pd(_mm256_extractf128_ps(b1, 1));
            let b2_hi = _mm_castps_pd(_mm256_extractf128_ps(b2, 1));
            let b3_hi = _mm_castps_pd(_mm256_extractf128_ps(b3, 1));

            let g0_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0_lo, b1_lo));
            let g0_23 = _mm_castpd_ps(_mm_unpacklo_pd(b2_lo, b3_lo));
            _mm256_storeu_ps(
                dst_ptr.add(g * 4).cast::<f32>(),
                _mm256_set_m128(g0_23, g0_01),
            );

            let g1_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0_lo, b1_lo));
            let g1_23 = _mm_castpd_ps(_mm_unpackhi_pd(b2_lo, b3_lo));
            _mm256_storeu_ps(
                dst_ptr.add((g + 1) * 4).cast::<f32>(),
                _mm256_set_m128(g1_23, g1_01),
            );

            let g2_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0_hi, b1_hi));
            let g2_23 = _mm_castpd_ps(_mm_unpacklo_pd(b2_hi, b3_hi));
            _mm256_storeu_ps(
                dst_ptr.add((g + 2) * 4).cast::<f32>(),
                _mm256_set_m128(g2_23, g2_01),
            );

            let g3_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0_hi, b1_hi));
            let g3_23 = _mm_castpd_ps(_mm_unpackhi_pd(b2_hi, b3_hi));
            _mm256_storeu_ps(
                dst_ptr.add((g + 3) * 4).cast::<f32>(),
                _mm256_set_m128(g3_23, g3_01),
            );

            g += 4;
        }
        // 2-group tail
        if g + 1 < g_count {
            let a0 = _mm_loadu_ps(src_ptr.add(g).cast::<f32>());
            let a1 = _mm_loadu_ps(src_ptr.add(stride + g).cast::<f32>());
            let a2 = _mm_loadu_ps(src_ptr.add(2 * stride + g).cast::<f32>());
            let a3 = _mm_loadu_ps(src_ptr.add(3 * stride + g).cast::<f32>());
            let (b0, b1, b2, b3) = dft4_f32_128::<INVERSE>(a0, a1, a2, a3);
            let b0d = _mm_castps_pd(b0);
            let b1d = _mm_castps_pd(b1);
            let b2d = _mm_castps_pd(b2);
            let b3d = _mm_castps_pd(b3);
            let g0_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0d, b1d));
            let g0_23 = _mm_castpd_ps(_mm_unpacklo_pd(b2d, b3d));
            _mm256_storeu_ps(
                dst_ptr.add(g * 4).cast::<f32>(),
                _mm256_set_m128(g0_23, g0_01),
            );
            let g1_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0d, b1d));
            let g1_23 = _mm_castpd_ps(_mm_unpackhi_pd(b2d, b3d));
            _mm256_storeu_ps(
                dst_ptr.add((g + 1) * 4).cast::<f32>(),
                _mm256_set_m128(g1_23, g1_01),
            );
            g += 2;
        }
        // Scalar tail
        while g < g_count {
            let a0s = *src_ptr.add(g);
            let a1s = *src_ptr.add(stride + g);
            let a2s = *src_ptr.add(2 * stride + g);
            let a3s = *src_ptr.add(3 * stride + g);
            let t0 = a0s + a2s;
            let t1 = a0s - a2s;
            let t2 = a1s + a3s;
            let t3 = a1s - a3s;
            let (it3_re, it3_im) = if INVERSE {
                (-t3.im, t3.re)
            } else {
                (t3.im, -t3.re)
            };
            let it3 = Complex {
                re: it3_re,
                im: it3_im,
            };
            let d = dst_ptr.add(g * 4);
            *d.add(0) = t0 + t2;
            *d.add(1) = t1 + it3;
            *d.add(2) = t0 - t2;
            *d.add(3) = t1 - it3;
            g += 1;
        }
    } else if prev_len >= 4 {
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;

            let mut j = 0usize;
            while j + 3 < prev_len {
                let off0 = src_base + j;
                let off1 = stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;

                let a0 = _mm256_loadu_ps(src_ptr.add(off0).cast::<f32>());
                let mut a1 = _mm256_loadu_ps(src_ptr.add(off1).cast::<f32>());
                let mut a2 = _mm256_loadu_ps(src_ptr.add(off2).cast::<f32>());
                let mut a3 = _mm256_loadu_ps(src_ptr.add(off3).cast::<f32>());

                let tw1 = _mm256_loadu_ps(tw_ptr.add(j).cast::<f32>());
                let tw2 = _mm256_loadu_ps(tw_ptr.add(prev_len + j).cast::<f32>());
                let tw3 = _mm256_loadu_ps(tw_ptr.add(2 * prev_len + j).cast::<f32>());
                if prev_len >= 16 {
                    _mm_prefetch(tw_ptr.add(j + 8).cast::<i8>(), _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(prev_len + j + 8).cast::<i8>(), _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(2 * prev_len + j + 8).cast::<i8>(), _MM_HINT_T0);
                }
                a1 = cmul_f32(a1, tw1);
                a2 = cmul_f32(a2, tw2);
                a3 = cmul_f32(a3, tw3);

                let (b0, b1, b2, b3) = dft4_f32::<INVERSE>(a0, a1, a2, a3);

                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_ps(dp.add(j).cast::<f32>(), b0);
                _mm256_storeu_ps(dp.add(j + prev_len).cast::<f32>(), b1);
                _mm256_storeu_ps(dp.add(j + 2 * prev_len).cast::<f32>(), b2);
                _mm256_storeu_ps(dp.add(j + 3 * prev_len).cast::<f32>(), b3);

                j += 4;
            }
            // 2-column tail
            while j + 1 < prev_len {
                let off0 = src_base + j;
                let off1 = stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;
                let a0 = _mm_loadu_ps(src_ptr.add(off0).cast::<f32>());
                let mut a1 = _mm_loadu_ps(src_ptr.add(off1).cast::<f32>());
                let mut a2 = _mm_loadu_ps(src_ptr.add(off2).cast::<f32>());
                let mut a3 = _mm_loadu_ps(src_ptr.add(off3).cast::<f32>());
                let tw1 = _mm_loadu_ps(tw_ptr.add(j).cast::<f32>());
                let tw2 = _mm_loadu_ps(tw_ptr.add(prev_len + j).cast::<f32>());
                let tw3 = _mm_loadu_ps(tw_ptr.add(2 * prev_len + j).cast::<f32>());
                a1 = cmul_f32_128(a1, tw1);
                a2 = cmul_f32_128(a2, tw2);
                a3 = cmul_f32_128(a3, tw3);
                let (b0, b1, b2, b3) = dft4_f32_128::<INVERSE>(a0, a1, a2, a3);
                let dp = dst_ptr.add(dst_base);
                _mm_storeu_ps(dp.add(j).cast::<f32>(), b0);
                _mm_storeu_ps(dp.add(j + prev_len).cast::<f32>(), b1);
                _mm_storeu_ps(dp.add(j + 2 * prev_len).cast::<f32>(), b2);
                _mm_storeu_ps(dp.add(j + 3 * prev_len).cast::<f32>(), b3);
                j += 2;
            }
            // Scalar tail
            if j < prev_len {
                let src_j = src_base + j;
                let a0s = *src_ptr.add(src_j);
                let mut a1s = *src_ptr.add(stride + src_j);
                let mut a2s = *src_ptr.add(2 * stride + src_j);
                let mut a3s = *src_ptr.add(3 * stride + src_j);
                let tw1 = *tw_ptr.add(j);
                let tw2 = *tw_ptr.add(prev_len + j);
                let tw3 = *tw_ptr.add(2 * prev_len + j);
                a1s = Complex {
                    re: a1s.re * tw1.re - a1s.im * tw1.im,
                    im: a1s.re * tw1.im + a1s.im * tw1.re,
                };
                a2s = Complex {
                    re: a2s.re * tw2.re - a2s.im * tw2.im,
                    im: a2s.re * tw2.im + a2s.im * tw2.re,
                };
                a3s = Complex {
                    re: a3s.re * tw3.re - a3s.im * tw3.im,
                    im: a3s.re * tw3.im + a3s.im * tw3.re,
                };
                let t0 = a0s + a2s;
                let t1 = a0s - a2s;
                let t2 = a1s + a3s;
                let t3 = a1s - a3s;
                let (it3_re, it3_im) = if INVERSE {
                    (-t3.im, t3.re)
                } else {
                    (t3.im, -t3.re)
                };
                let it3 = Complex {
                    re: it3_re,
                    im: it3_im,
                };
                let dp = dst_ptr.add(dst_base);
                *dp.add(j) = t0 + t2;
                *dp.add(j + prev_len) = t1 + it3;
                *dp.add(j + 2 * prev_len) = t0 - t2;
                *dp.add(j + 3 * prev_len) = t1 - it3;
            }
        }
    } else {
        // prev_len in {2, 3}: 2-column __m128 path.
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;
            while j + 1 < prev_len {
                let off0 = src_base + j;
                let off1 = stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let off3 = 3 * stride + src_base + j;
                let a0 = _mm_loadu_ps(src_ptr.add(off0).cast::<f32>());
                let mut a1 = _mm_loadu_ps(src_ptr.add(off1).cast::<f32>());
                let mut a2 = _mm_loadu_ps(src_ptr.add(off2).cast::<f32>());
                let mut a3 = _mm_loadu_ps(src_ptr.add(off3).cast::<f32>());
                let tw1 = _mm_loadu_ps(tw.as_ptr().add(j).cast::<f32>());
                let tw2 = _mm_loadu_ps(tw.as_ptr().add(prev_len + j).cast::<f32>());
                let tw3 = _mm_loadu_ps(tw.as_ptr().add(2 * prev_len + j).cast::<f32>());
                a1 = cmul_f32_128(a1, tw1);
                a2 = cmul_f32_128(a2, tw2);
                a3 = cmul_f32_128(a3, tw3);
                let (b0, b1, b2, b3) = dft4_f32_128::<INVERSE>(a0, a1, a2, a3);
                let dp = dst_ptr.add(dst_base);
                _mm_storeu_ps(dp.add(j).cast::<f32>(), b0);
                _mm_storeu_ps(dp.add(j + prev_len).cast::<f32>(), b1);
                _mm_storeu_ps(dp.add(j + 2 * prev_len).cast::<f32>(), b2);
                _mm_storeu_ps(dp.add(j + 3 * prev_len).cast::<f32>(), b3);
                j += 2;
            }
            if j < prev_len {
                let src_j = src_base + j;
                let a0s = *src_ptr.add(src_j);
                let mut a1s = *src_ptr.add(stride + src_j);
                let mut a2s = *src_ptr.add(2 * stride + src_j);
                let mut a3s = *src_ptr.add(3 * stride + src_j);
                if j > 0 {
                    let tw1 = *tw.as_ptr().add(j);
                    let tw2 = *tw.as_ptr().add(prev_len + j);
                    let tw3 = *tw.as_ptr().add(2 * prev_len + j);
                    a1s = Complex {
                        re: a1s.re * tw1.re - a1s.im * tw1.im,
                        im: a1s.re * tw1.im + a1s.im * tw1.re,
                    };
                    a2s = Complex {
                        re: a2s.re * tw2.re - a2s.im * tw2.im,
                        im: a2s.re * tw2.im + a2s.im * tw2.re,
                    };
                    a3s = Complex {
                        re: a3s.re * tw3.re - a3s.im * tw3.im,
                        im: a3s.re * tw3.im + a3s.im * tw3.re,
                    };
                }
                let t0 = a0s + a2s;
                let t1 = a0s - a2s;
                let t2 = a1s + a3s;
                let t3 = a1s - a3s;
                let (it3_re, it3_im) = if INVERSE {
                    (-t3.im, t3.re)
                } else {
                    (t3.im, -t3.re)
                };
                let it3 = Complex {
                    re: it3_re,
                    im: it3_im,
                };
                let dp = dst_ptr.add(dst_base);
                *dp.add(j) = t0 + t2;
                *dp.add(j + prev_len) = t1 + it3;
                *dp.add(j + 2 * prev_len) = t0 - t2;
                *dp.add(j + 3 * prev_len) = t1 - it3;
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f32(dst_ptr, pw.as_ptr(), dst.len());
    }
}
