//! AVX2+FMA flat radix-3 Stockham pass for f64 and f32.
//!
//! DFT-3 butterfly (from dft3_impl):
//!   sum = x1 + x2,  diff = x1 - x2
//!   b0  = x0 + sum
//!   m0  = x0 + sum * (-0.5)          [fused: fmadd(sum, -0.5, x0)]
//!   m1  = rot_neg_i(diff) * s        [forward;  s = √3/2]
//!         rot_pos_i(diff) * s        [inverse]
//!   b1  = m0 + m1
//!   b2  = m0 - m1
//!
//! Cost: 4 real multiplies, 6 complex adds — minimal for radix-3.

use super::helpers::{
    apply_pointwise_f32, apply_pointwise_f64, cmul, cmul_f32, cmul_f32_128, rot_neg_i,
    rot_neg_i_f32, rot_pos_i, rot_pos_i_f32,
};
use num_complex::Complex;
use std::arch::x86_64::{
    _mm256_add_pd, _mm256_add_ps, _mm256_castpd256_pd128, _mm256_castps256_ps128,
    _mm256_extractf128_pd, _mm256_extractf128_ps, _mm256_fmadd_pd, _mm256_fmadd_ps,
    _mm256_loadu_pd, _mm256_loadu_ps, _mm256_mul_pd, _mm256_mul_ps, _mm256_set1_pd, _mm256_set1_ps,
    _mm256_set_m128d, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_sub_pd, _mm256_sub_ps, _mm_add_ps,
    _mm_castpd_ps, _mm_castps_pd, _mm_fmadd_ps, _mm_loadu_ps, _mm_mul_ps, _mm_permute_ps,
    _mm_prefetch, _mm_set1_ps, _mm_set_ps, _mm_store_sd, _mm_storeu_pd, _mm_storeu_ps, _mm_sub_ps,
    _mm_unpackhi_pd, _mm_unpacklo_pd, _mm_xor_ps, _MM_HINT_T0,
};

// ── f64 radix-3 ─────────────────────────────────────────────────────────────

/// Scalar DFT-3 butterfly: reads a0,a1,a2 and writes b0,b1,b2 to `out[0..3]`.
#[inline]
unsafe fn scalar_dft3<const INVERSE: bool>(
    a0: Complex<f64>,
    a1: Complex<f64>,
    a2: Complex<f64>,
    out: *mut Complex<f64>,
) {
    let s: f64 = 0.8660254037844386;
    let wr: f64 = -0.5;
    let sum_re = a1.re + a2.re;
    let sum_im = a1.im + a2.im;
    let diff_re = a1.re - a2.re;
    let diff_im = a1.im - a2.im;
    let m0_re = a0.re + sum_re * wr;
    let m0_im = a0.im + sum_im * wr;
    let (m1_re, m1_im) = if INVERSE {
        (-diff_im * s, diff_re * s)
    } else {
        (diff_im * s, -diff_re * s)
    };
    *out.add(0) = Complex {
        re: a0.re + sum_re,
        im: a0.im + sum_im,
    };
    *out.add(1) = Complex {
        re: m0_re + m1_re,
        im: m0_im + m1_im,
    };
    *out.add(2) = Complex {
        re: m0_re - m1_re,
        im: m0_im - m1_im,
    };
}

/// Flat radix-3 Stockham pass in AVX2+FMA for f64. Processes all `g_count` groups.
///
/// `tw` has `(R-1) × prev_len = 2 × prev_len` entries:
///   arm 1 at `[0*prev_len .. 1*prev_len)`, arm 2 at `[1*prev_len .. 2*prev_len)`.
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r3_f64<
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
    let stride = g_count * prev_len; // = n/3 for r=3
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        // No twiddles (W^0 = 1). 2 groups per AVX2 iteration.
        let mut g = 0usize;
        while g + 1 < g_count {
            let a0 = _mm256_loadu_pd(src_ptr.add(g).cast::<f64>());
            let a1 = _mm256_loadu_pd(src_ptr.add(stride + g).cast::<f64>());
            let a2 = _mm256_loadu_pd(src_ptr.add(2 * stride + g).cast::<f64>());

            let wr = _mm256_set1_pd(-0.5);
            let s = _mm256_set1_pd(0.8660254037844386_f64);
            let sum = _mm256_add_pd(a1, a2);
            let diff = _mm256_sub_pd(a1, a2);
            let b0 = _mm256_add_pd(a0, sum);
            let m0 = _mm256_fmadd_pd(sum, wr, a0);
            let m1_dir = if INVERSE {
                rot_pos_i(diff)
            } else {
                rot_neg_i(diff)
            };
            let m1 = _mm256_mul_pd(m1_dir, s);
            let b1 = _mm256_add_pd(m0, m1);
            let b2 = _mm256_sub_pd(m0, m1);

            // b0 = [b0g0, b0g1], b1 = [b1g0, b1g1], b2 = [b2g0, b2g1].
            let b0_lo = _mm256_castpd256_pd128(b0);
            let b1_lo = _mm256_castpd256_pd128(b1);
            let b2_lo = _mm256_castpd256_pd128(b2);
            let b0_hi = _mm256_extractf128_pd(b0, 1);
            let b1_hi = _mm256_extractf128_pd(b1, 1);
            let b2_hi = _mm256_extractf128_pd(b2, 1);

            // Group g
            let dg = dst_ptr.add(g * 3);
            _mm256_storeu_pd(dg.cast::<f64>(), _mm256_set_m128d(b1_lo, b0_lo));
            _mm_storeu_pd(dg.add(2).cast::<f64>(), b2_lo);

            // Group g+1
            let dg1 = dst_ptr.add((g + 1) * 3);
            _mm256_storeu_pd(dg1.cast::<f64>(), _mm256_set_m128d(b1_hi, b0_hi));
            _mm_storeu_pd(dg1.add(2).cast::<f64>(), b2_hi);

            g += 2;
        }
        // Scalar tail
        if g < g_count {
            let a0s = *src_ptr.add(g);
            let a1s = *src_ptr.add(stride + g);
            let a2s = *src_ptr.add(2 * stride + g);
            scalar_dft3::<INVERSE>(a0s, a1s, a2s, dst_ptr.add(g * 3));
        }
    } else {
        // prev_len >= 2: twiddles applied.
        let wr = _mm256_set1_pd(-0.5);
        let s = _mm256_set1_pd(0.8660254037844386_f64);
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;

            // Width-4 loop
            while j + 3 < prev_len {
                let off0a = src_base + j;
                let off1a = stride + src_base + j;
                let off2a = 2 * stride + src_base + j;
                let a0_lo = _mm256_loadu_pd(src_ptr.add(off0a).cast::<f64>());
                let a1_lo = _mm256_loadu_pd(src_ptr.add(off1a).cast::<f64>());
                let a2_lo = _mm256_loadu_pd(src_ptr.add(off2a).cast::<f64>());
                let a0_hi = _mm256_loadu_pd(src_ptr.add(off0a + 2).cast::<f64>());
                let a1_hi_raw = _mm256_loadu_pd(src_ptr.add(off1a + 2).cast::<f64>());
                let a2_hi_raw = _mm256_loadu_pd(src_ptr.add(off2a + 2).cast::<f64>());
                let tw1_lo = _mm256_loadu_pd(tw_ptr.add(j).cast::<f64>());
                let tw2_lo = _mm256_loadu_pd(tw_ptr.add(prev_len + j).cast::<f64>());
                let tw1_hi = _mm256_loadu_pd(tw_ptr.add(j + 2).cast::<f64>());
                let tw2_hi = _mm256_loadu_pd(tw_ptr.add(prev_len + j + 2).cast::<f64>());
                if prev_len >= 16 {
                    _mm_prefetch(tw_ptr.add(j + 8).cast::<i8>(), _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(prev_len + j + 8).cast::<i8>(), _MM_HINT_T0);
                }
                let a1_lo_tw = cmul(a1_lo, tw1_lo);
                let a2_lo_tw = cmul(a2_lo, tw2_lo);
                let a1_hi = cmul(a1_hi_raw, tw1_hi);
                let a2_hi = cmul(a2_hi_raw, tw2_hi);
                // DFT-3 chain lo
                let sum_lo = _mm256_add_pd(a1_lo_tw, a2_lo_tw);
                let diff_lo = _mm256_sub_pd(a1_lo_tw, a2_lo_tw);
                let b0_lo = _mm256_add_pd(a0_lo, sum_lo);
                let m0_lo = _mm256_fmadd_pd(sum_lo, wr, a0_lo);
                let m1_dir_lo = if INVERSE {
                    rot_pos_i(diff_lo)
                } else {
                    rot_neg_i(diff_lo)
                };
                let m1_lo = _mm256_mul_pd(m1_dir_lo, s);
                let b1_lo = _mm256_add_pd(m0_lo, m1_lo);
                let b2_lo = _mm256_sub_pd(m0_lo, m1_lo);
                // DFT-3 chain hi
                let sum_hi = _mm256_add_pd(a1_hi, a2_hi);
                let diff_hi = _mm256_sub_pd(a1_hi, a2_hi);
                let b0_hi = _mm256_add_pd(a0_hi, sum_hi);
                let m0_hi = _mm256_fmadd_pd(sum_hi, wr, a0_hi);
                let m1_dir_hi = if INVERSE {
                    rot_pos_i(diff_hi)
                } else {
                    rot_neg_i(diff_hi)
                };
                let m1_hi = _mm256_mul_pd(m1_dir_hi, s);
                let b1_hi = _mm256_add_pd(m0_hi, m1_hi);
                let b2_hi = _mm256_sub_pd(m0_hi, m1_hi);
                // Stores
                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_pd(dp.add(j).cast::<f64>(), b0_lo);
                _mm256_storeu_pd(dp.add(j + 2).cast::<f64>(), b0_hi);
                _mm256_storeu_pd(dp.add(j + prev_len).cast::<f64>(), b1_lo);
                _mm256_storeu_pd(dp.add(j + 2 + prev_len).cast::<f64>(), b1_hi);
                _mm256_storeu_pd(dp.add(j + 2 * prev_len).cast::<f64>(), b2_lo);
                _mm256_storeu_pd(dp.add(j + 2 + 2 * prev_len).cast::<f64>(), b2_hi);
                j += 4;
            }

            // Width-2 loop
            while j + 1 < prev_len {
                let off0 = src_base + j;
                let off1 = stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let a0 = _mm256_loadu_pd(src_ptr.add(off0).cast::<f64>());
                let a1_raw = _mm256_loadu_pd(src_ptr.add(off1).cast::<f64>());
                let a2_raw = _mm256_loadu_pd(src_ptr.add(off2).cast::<f64>());
                let tw1 = _mm256_loadu_pd(tw_ptr.add(j).cast::<f64>());
                let tw2 = _mm256_loadu_pd(tw_ptr.add(prev_len + j).cast::<f64>());
                let a1 = cmul(a1_raw, tw1);
                let a2 = cmul(a2_raw, tw2);

                let sum = _mm256_add_pd(a1, a2);
                let diff = _mm256_sub_pd(a1, a2);
                let b0 = _mm256_add_pd(a0, sum);
                let m0 = _mm256_fmadd_pd(sum, wr, a0);
                let m1_dir = if INVERSE {
                    rot_pos_i(diff)
                } else {
                    rot_neg_i(diff)
                };
                let m1 = _mm256_mul_pd(m1_dir, s);
                let b1 = _mm256_add_pd(m0, m1);
                let b2 = _mm256_sub_pd(m0, m1);

                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_pd(dp.add(j).cast::<f64>(), b0);
                _mm256_storeu_pd(dp.add(j + prev_len).cast::<f64>(), b1);
                _mm256_storeu_pd(dp.add(j + 2 * prev_len).cast::<f64>(), b2);
                j += 2;
            }
            if j < prev_len {
                let src_j = src_base + j;
                let a0s = *src_ptr.add(src_j);
                let mut a1s = *src_ptr.add(stride + src_j);
                let mut a2s = *src_ptr.add(2 * stride + src_j);
                if j > 0 {
                    let tw1 = *tw.as_ptr().add(j);
                    let tw2 = *tw.as_ptr().add(prev_len + j);
                    a1s = Complex {
                        re: a1s.re * tw1.re - a1s.im * tw1.im,
                        im: a1s.re * tw1.im + a1s.im * tw1.re,
                    };
                    a2s = Complex {
                        re: a2s.re * tw2.re - a2s.im * tw2.im,
                        im: a2s.re * tw2.im + a2s.im * tw2.re,
                    };
                }
                let dp = dst_ptr.add(dst_base);
                let s: f64 = 0.8660254037844386;
                let wr: f64 = -0.5;
                let sum_re = a1s.re + a2s.re;
                let sum_im = a1s.im + a2s.im;
                let diff_re = a1s.re - a2s.re;
                let diff_im = a1s.im - a2s.im;
                let m0_re = a0s.re + sum_re * wr;
                let m0_im = a0s.im + sum_im * wr;
                let (m1_re, m1_im) = if INVERSE {
                    (-diff_im * s, diff_re * s)
                } else {
                    (diff_im * s, -diff_re * s)
                };
                *dp.add(j) = Complex {
                    re: a0s.re + sum_re,
                    im: a0s.im + sum_im,
                };
                *dp.add(j + prev_len) = Complex {
                    re: m0_re + m1_re,
                    im: m0_im + m1_im,
                };
                *dp.add(j + 2 * prev_len) = Complex {
                    re: m0_re - m1_re,
                    im: m0_im - m1_im,
                };
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f64(dst_ptr, pw.as_ptr(), dst.len());
    }
}

// ── f32 radix-3 ─────────────────────────────────────────────────────────────

/// Scalar DFT-3 for f32, sequential output (prev_len==1 case).
#[inline]
unsafe fn scalar_dft3_f32<const INVERSE: bool>(
    a0: Complex<f32>,
    a1: Complex<f32>,
    a2: Complex<f32>,
    out: *mut Complex<f32>,
) {
    let s: f32 = 0.8660254_f32;
    let wr: f32 = -0.5_f32;
    let sum_re = a1.re + a2.re;
    let sum_im = a1.im + a2.im;
    let diff_re = a1.re - a2.re;
    let diff_im = a1.im - a2.im;
    let m0_re = a0.re + sum_re * wr;
    let m0_im = a0.im + sum_im * wr;
    let (m1_re, m1_im) = if INVERSE {
        (-diff_im * s, diff_re * s)
    } else {
        (diff_im * s, -diff_re * s)
    };
    *out.add(0) = Complex {
        re: a0.re + sum_re,
        im: a0.im + sum_im,
    };
    *out.add(1) = Complex {
        re: m0_re + m1_re,
        im: m0_im + m1_im,
    };
    *out.add(2) = Complex {
        re: m0_re - m1_re,
        im: m0_im - m1_im,
    };
}

/// Scalar DFT-3 column tail for f32, column-major output (prev_len>=2 case).
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn scalar_dft3_f32_col<const INVERSE: bool>(
    src_ptr: *const Complex<f32>,
    dst_ptr: *mut Complex<f32>,
    stride: usize,
    dst_base: usize,
    src_base: usize,
    j: usize,
    prev_len: usize,
    tw: &[Complex<f32>],
) {
    let src_j = src_base + j;
    let a0s = *src_ptr.add(src_j);
    let mut a1s = *src_ptr.add(stride + src_j);
    let mut a2s = *src_ptr.add(2 * stride + src_j);
    if j > 0 {
        let tw1 = *tw.as_ptr().add(j);
        let tw2 = *tw.as_ptr().add(prev_len + j);
        a1s = Complex {
            re: a1s.re * tw1.re - a1s.im * tw1.im,
            im: a1s.re * tw1.im + a1s.im * tw1.re,
        };
        a2s = Complex {
            re: a2s.re * tw2.re - a2s.im * tw2.im,
            im: a2s.re * tw2.im + a2s.im * tw2.re,
        };
    }
    let s: f32 = 0.8660254_f32;
    let wr: f32 = -0.5_f32;
    let sum_re = a1s.re + a2s.re;
    let sum_im = a1s.im + a2s.im;
    let diff_re = a1s.re - a2s.re;
    let diff_im = a1s.im - a2s.im;
    let m0_re = a0s.re + sum_re * wr;
    let m0_im = a0s.im + sum_im * wr;
    let (m1_re, m1_im) = if INVERSE {
        (-diff_im * s, diff_re * s)
    } else {
        (diff_im * s, -diff_re * s)
    };
    let dp = dst_ptr.add(dst_base);
    *dp.add(j) = Complex {
        re: a0s.re + sum_re,
        im: a0s.im + sum_im,
    };
    *dp.add(j + prev_len) = Complex {
        re: m0_re + m1_re,
        im: m0_im + m1_im,
    };
    *dp.add(j + 2 * prev_len) = Complex {
        re: m0_re - m1_re,
        im: m0_im - m1_im,
    };
}

/// Flat radix-3 Stockham pass in AVX2+FMA for f32. Processes all `g_count` groups.
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r3_f32<
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
    let stride = g_count * prev_len;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let wr_f32 = _mm256_set1_ps(-0.5_f32);
    let s_f32 = _mm256_set1_ps(0.8660254037844386_f32);
    let wr_128 = _mm_set1_ps(-0.5_f32);
    let s_128 = _mm_set1_ps(0.8660254037844386_f32);

    // DFT3 for 4 Complex<f32> via __m256. Returns (b0, b1, b2).
    macro_rules! dft3_256 {
        ($a0:expr, $a1:expr, $a2:expr) => {{
            let sum = _mm256_add_ps($a1, $a2);
            let diff = _mm256_sub_ps($a1, $a2);
            let b0 = _mm256_add_ps($a0, sum);
            let m0 = _mm256_fmadd_ps(sum, wr_f32, $a0);
            let m1 = _mm256_mul_ps(
                if INVERSE {
                    rot_pos_i_f32(diff)
                } else {
                    rot_neg_i_f32(diff)
                },
                s_f32,
            );
            (b0, _mm256_add_ps(m0, m1), _mm256_sub_ps(m0, m1))
        }};
    }

    // DFT3 for 2 Complex<f32> via __m128.
    macro_rules! dft3_128 {
        ($a0:expr, $a1:expr, $a2:expr) => {{
            let sum = _mm_add_ps($a1, $a2);
            let diff = _mm_sub_ps($a1, $a2);
            let b0 = _mm_add_ps($a0, sum);
            let m0 = _mm_fmadd_ps(sum, wr_128, $a0);
            let perm = _mm_permute_ps(diff, 0xB1);
            let sign_fwd = _mm_set_ps(-0.0, 0.0, -0.0, 0.0);
            let sign_inv = _mm_set_ps(0.0, -0.0, 0.0, -0.0);
            let m1_dir = _mm_xor_ps(perm, if INVERSE { sign_inv } else { sign_fwd });
            let m1 = _mm_mul_ps(m1_dir, s_128);
            (b0, _mm_add_ps(m0, m1), _mm_sub_ps(m0, m1))
        }};
    }

    if prev_len == 1 {
        let mut g = 0usize;
        while g + 3 < g_count {
            let a0 = _mm256_loadu_ps(src_ptr.add(g).cast::<f32>());
            let a1 = _mm256_loadu_ps(src_ptr.add(stride + g).cast::<f32>());
            let a2 = _mm256_loadu_ps(src_ptr.add(2 * stride + g).cast::<f32>());
            let (b0, b1, b2) = dft3_256!(a0, a1, a2);

            let b0_lo_d = _mm_castps_pd(_mm256_castps256_ps128(b0));
            let b1_lo_d = _mm_castps_pd(_mm256_castps256_ps128(b1));
            let b2_lo_d = _mm_castps_pd(_mm256_castps256_ps128(b2));
            let b0_hi_d = _mm_castps_pd(_mm256_extractf128_ps(b0, 1));
            let b1_hi_d = _mm_castps_pd(_mm256_extractf128_ps(b1, 1));
            let b2_hi_d = _mm_castps_pd(_mm256_extractf128_ps(b2, 1));

            let g0_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0_lo_d, b1_lo_d));
            _mm_storeu_ps(dst_ptr.add(g * 3).cast::<f32>(), g0_01);
            _mm_store_sd(dst_ptr.add(g * 3 + 2).cast::<f64>(), b2_lo_d);

            let g1_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0_lo_d, b1_lo_d));
            _mm_storeu_ps(dst_ptr.add((g + 1) * 3).cast::<f32>(), g1_01);
            _mm_store_sd(
                dst_ptr.add((g + 1) * 3 + 2).cast::<f64>(),
                _mm_unpackhi_pd(b2_lo_d, b2_lo_d),
            );

            let g2_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0_hi_d, b1_hi_d));
            _mm_storeu_ps(dst_ptr.add((g + 2) * 3).cast::<f32>(), g2_01);
            _mm_store_sd(dst_ptr.add((g + 2) * 3 + 2).cast::<f64>(), b2_hi_d);

            let g3_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0_hi_d, b1_hi_d));
            _mm_storeu_ps(dst_ptr.add((g + 3) * 3).cast::<f32>(), g3_01);
            _mm_store_sd(
                dst_ptr.add((g + 3) * 3 + 2).cast::<f64>(),
                _mm_unpackhi_pd(b2_hi_d, b2_hi_d),
            );

            g += 4;
        }
        while g + 1 < g_count {
            let a0 = _mm_loadu_ps(src_ptr.add(g).cast::<f32>());
            let a1 = _mm_loadu_ps(src_ptr.add(stride + g).cast::<f32>());
            let a2 = _mm_loadu_ps(src_ptr.add(2 * stride + g).cast::<f32>());
            let (b0, b1, b2) = dft3_128!(a0, a1, a2);
            let b0d = _mm_castps_pd(b0);
            let b1d = _mm_castps_pd(b1);
            let b2d = _mm_castps_pd(b2);
            let g0_01 = _mm_castpd_ps(_mm_unpacklo_pd(b0d, b1d));
            _mm_storeu_ps(dst_ptr.add(g * 3).cast::<f32>(), g0_01);
            _mm_store_sd(dst_ptr.add(g * 3 + 2).cast::<f64>(), b2d);
            let g1_01 = _mm_castpd_ps(_mm_unpackhi_pd(b0d, b1d));
            _mm_storeu_ps(dst_ptr.add((g + 1) * 3).cast::<f32>(), g1_01);
            _mm_store_sd(
                dst_ptr.add((g + 1) * 3 + 2).cast::<f64>(),
                _mm_unpackhi_pd(b2d, b2d),
            );
            g += 2;
        }
        while g < g_count {
            let a0s = *src_ptr.add(g);
            let a1s = *src_ptr.add(stride + g);
            let a2s = *src_ptr.add(2 * stride + g);
            scalar_dft3_f32::<INVERSE>(a0s, a1s, a2s, dst_ptr.add(g * 3));
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
                let a0 = _mm256_loadu_ps(src_ptr.add(off0).cast::<f32>());
                let tw1 = _mm256_loadu_ps(tw_ptr.add(j).cast::<f32>());
                let tw2 = _mm256_loadu_ps(tw_ptr.add(prev_len + j).cast::<f32>());
                let a1 = cmul_f32(_mm256_loadu_ps(src_ptr.add(off1).cast::<f32>()), tw1);
                let a2 = cmul_f32(_mm256_loadu_ps(src_ptr.add(off2).cast::<f32>()), tw2);
                if prev_len >= 16 {
                    _mm_prefetch(tw_ptr.add(j + 8).cast::<i8>(), _MM_HINT_T0);
                    _mm_prefetch(tw_ptr.add(prev_len + j + 8).cast::<i8>(), _MM_HINT_T0);
                }
                let (b0, b1, b2) = dft3_256!(a0, a1, a2);
                let dp = dst_ptr.add(dst_base);
                _mm256_storeu_ps(dp.add(j).cast::<f32>(), b0);
                _mm256_storeu_ps(dp.add(j + prev_len).cast::<f32>(), b1);
                _mm256_storeu_ps(dp.add(j + 2 * prev_len).cast::<f32>(), b2);
                j += 4;
            }
            while j + 1 < prev_len {
                let off0 = src_base + j;
                let off1 = stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let tw1 = _mm_loadu_ps(tw_ptr.add(j).cast::<f32>());
                let tw2 = _mm_loadu_ps(tw_ptr.add(prev_len + j).cast::<f32>());
                let a0 = _mm_loadu_ps(src_ptr.add(off0).cast::<f32>());
                let a1 = cmul_f32_128(_mm_loadu_ps(src_ptr.add(off1).cast::<f32>()), tw1);
                let a2 = cmul_f32_128(_mm_loadu_ps(src_ptr.add(off2).cast::<f32>()), tw2);
                let (b0, b1, b2) = dft3_128!(a0, a1, a2);
                let dp = dst_ptr.add(dst_base);
                _mm_storeu_ps(dp.add(j).cast::<f32>(), b0);
                _mm_storeu_ps(dp.add(j + prev_len).cast::<f32>(), b1);
                _mm_storeu_ps(dp.add(j + 2 * prev_len).cast::<f32>(), b2);
                j += 2;
            }
            if j < prev_len {
                scalar_dft3_f32_col::<INVERSE>(
                    src_ptr, dst_ptr, stride, dst_base, src_base, j, prev_len, tw,
                );
            }
        }
    } else {
        // prev_len in {2, 3}: 2-column __m128 loop.
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;
            while j + 1 < prev_len {
                let off0 = src_base + j;
                let off1 = stride + src_base + j;
                let off2 = 2 * stride + src_base + j;
                let tw1 = _mm_loadu_ps(tw_ptr.add(j).cast::<f32>());
                let tw2 = _mm_loadu_ps(tw_ptr.add(prev_len + j).cast::<f32>());
                let a0 = _mm_loadu_ps(src_ptr.add(off0).cast::<f32>());
                let a1 = cmul_f32_128(_mm_loadu_ps(src_ptr.add(off1).cast::<f32>()), tw1);
                let a2 = cmul_f32_128(_mm_loadu_ps(src_ptr.add(off2).cast::<f32>()), tw2);
                let (b0, b1, b2) = dft3_128!(a0, a1, a2);
                let dp = dst_ptr.add(dst_base);
                _mm_storeu_ps(dp.add(j).cast::<f32>(), b0);
                _mm_storeu_ps(dp.add(j + prev_len).cast::<f32>(), b1);
                _mm_storeu_ps(dp.add(j + 2 * prev_len).cast::<f32>(), b2);
                j += 2;
            }
            if j < prev_len {
                scalar_dft3_f32_col::<INVERSE>(
                    src_ptr, dst_ptr, stride, dst_base, src_base, j, prev_len, tw,
                );
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f32(dst_ptr, pw.as_ptr(), dst.len());
    }
}
