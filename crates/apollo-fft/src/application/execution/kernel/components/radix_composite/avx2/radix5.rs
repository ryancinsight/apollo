//! AVX2+FMA flat radix-5 Stockham pass for f64 and f32.
//!
//! Includes DFT-5 constants, scalar butterflies, and AVX2 vectorized butterflies.

use super::helpers::{
    apply_pointwise_f32, apply_pointwise_f64, cmul, cmul_f32, rot_pos_i, rot_pos_i_f32,
};
use eunomia::Complex;
use std::arch::x86_64::{
    __m256, __m256d, _mm256_add_pd, _mm256_add_ps, _mm256_fmadd_pd, _mm256_fmadd_ps,
    _mm256_fmsub_pd, _mm256_fmsub_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_mul_pd,
    _mm256_mul_ps, _mm256_set1_pd, _mm256_set1_ps, _mm256_storeu_pd, _mm256_storeu_ps,
    _mm256_sub_pd, _mm256_sub_ps,
};

// ── f64 radix-5 ─────────────────────────────────────────────────────────────

/// Radix-5 DFT constants: cos(2pi/5), cos(4pi/5), sin(2pi/5), sin(4pi/5).
const DFT5_C1: f64 = 0.309_016_994_374_947_45;
const DFT5_C2: f64 = -0.809_016_994_374_947_5;
const DFT5_S1: f64 = 0.951_056_516_295_153_5;
const DFT5_S2: f64 = 0.587_785_252_292_473_1;

/// Scalar radix-5 butterfly mirroring `winograd::radix::dft5_values`.
/// Writes the 5 outputs to `out[k*prev_len]`, k in 0..5. `s1`/`s2` carry the
/// direction sign (negated for forward); the imaginary rotation is fixed `+i`.
#[inline]
unsafe fn scalar_dft5<const INVERSE: bool>(
    a0: Complex<f64>,
    a1: Complex<f64>,
    a2: Complex<f64>,
    a3: Complex<f64>,
    a4: Complex<f64>,
    out: *mut Complex<f64>,
    prev_len: usize,
) {
    let (s1, s2) = if INVERSE {
        (DFT5_S1, DFT5_S2)
    } else {
        (-DFT5_S1, -DFT5_S2)
    };
    let t1_re = a1.re + a4.re;
    let t1_im = a1.im + a4.im;
    let t2_re = a1.re - a4.re;
    let t2_im = a1.im - a4.im;
    let t3_re = a2.re + a3.re;
    let t3_im = a2.im + a3.im;
    let t4_re = a2.re - a3.re;
    let t4_im = a2.im - a3.im;
    let m1_re = t1_re * DFT5_C1 + t3_re * DFT5_C2;
    let m1_im = t1_im * DFT5_C1 + t3_im * DFT5_C2;
    let m2_re = t1_re * DFT5_C2 + t3_re * DFT5_C1;
    let m2_im = t1_im * DFT5_C2 + t3_im * DFT5_C1;
    let q3_re = t2_re * s1 + t4_re * s2;
    let q3_im = t2_im * s1 + t4_im * s2;
    let q4_re = t2_re * s2 - t4_re * s1;
    let q4_im = t2_im * s2 - t4_im * s1;
    let a1c_re = a0.re + m1_re;
    let a1c_im = a0.im + m1_im;
    let a2c_re = a0.re + m2_re;
    let a2c_im = a0.im + m2_im;
    *out.add(0) = Complex::new(a0.re + t1_re + t3_re, a0.im + t1_im + t3_im);
    *out.add(prev_len) = Complex::new(a1c_re - q3_im, a1c_im + q3_re);
    *out.add(2 * prev_len) = Complex::new(a2c_re - q4_im, a2c_im + q4_re);
    *out.add(3 * prev_len) = Complex::new(a2c_re + q4_im, a2c_im - q4_re);
    *out.add(4 * prev_len) = Complex::new(a1c_re + q3_im, a1c_im - q3_re);
}

/// AVX2 radix-5 butterfly for 2 complex-f64 columns simultaneously.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft5_f64<const INVERSE: bool>(
    a0: __m256d,
    a1: __m256d,
    a2: __m256d,
    a3: __m256d,
    a4: __m256d,
) -> (__m256d, __m256d, __m256d, __m256d, __m256d) {
    let c1 = _mm256_set1_pd(DFT5_C1);
    let c2 = _mm256_set1_pd(DFT5_C2);
    let (s1, s2) = if INVERSE {
        (_mm256_set1_pd(DFT5_S1), _mm256_set1_pd(DFT5_S2))
    } else {
        (_mm256_set1_pd(-DFT5_S1), _mm256_set1_pd(-DFT5_S2))
    };
    let t1 = _mm256_add_pd(a1, a4);
    let t2 = _mm256_sub_pd(a1, a4);
    let t3 = _mm256_add_pd(a2, a3);
    let t4 = _mm256_sub_pd(a2, a3);
    let m1 = _mm256_fmadd_pd(t3, c2, _mm256_mul_pd(t1, c1));
    let m2 = _mm256_fmadd_pd(t3, c1, _mm256_mul_pd(t1, c2));
    let q3 = _mm256_fmadd_pd(t4, s2, _mm256_mul_pd(t2, s1));
    let q4 = _mm256_fmsub_pd(t2, s2, _mm256_mul_pd(t4, s1));
    let a1c = _mm256_add_pd(a0, m1);
    let a2c = _mm256_add_pd(a0, m2);
    let iq3 = rot_pos_i(q3);
    let iq4 = rot_pos_i(q4);
    (
        _mm256_add_pd(a0, _mm256_add_pd(t1, t3)),
        _mm256_add_pd(a1c, iq3),
        _mm256_add_pd(a2c, iq4),
        _mm256_sub_pd(a2c, iq4),
        _mm256_sub_pd(a1c, iq3),
    )
}

/// Flat radix-5 Stockham pass in AVX2+FMA for f64. Processes all `g_count` groups.
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r5_f64<
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
    let stride = g_count * prev_len; // n/5
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        for g in 0..g_count {
            let a0 = *src_ptr.add(g);
            let a1 = *src_ptr.add(stride + g);
            let a2 = *src_ptr.add(2 * stride + g);
            let a3 = *src_ptr.add(3 * stride + g);
            let a4 = *src_ptr.add(4 * stride + g);
            scalar_dft5::<INVERSE>(a0, a1, a2, a3, a4, dst_ptr.add(g * 5), 1);
        }
    } else {
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;
            while j + 1 < prev_len {
                let a0 = _mm256_loadu_pd(src_ptr.add(src_base + j).cast::<f64>());
                let a1r = _mm256_loadu_pd(src_ptr.add(stride + src_base + j).cast::<f64>());
                let a2r = _mm256_loadu_pd(src_ptr.add(2 * stride + src_base + j).cast::<f64>());
                let a3r = _mm256_loadu_pd(src_ptr.add(3 * stride + src_base + j).cast::<f64>());
                let a4r = _mm256_loadu_pd(src_ptr.add(4 * stride + src_base + j).cast::<f64>());
                let a1 = cmul(a1r, _mm256_loadu_pd(tw_ptr.add(j).cast::<f64>()));
                let a2 = cmul(a2r, _mm256_loadu_pd(tw_ptr.add(prev_len + j).cast::<f64>()));
                let a3 = cmul(
                    a3r,
                    _mm256_loadu_pd(tw_ptr.add(2 * prev_len + j).cast::<f64>()),
                );
                let a4 = cmul(
                    a4r,
                    _mm256_loadu_pd(tw_ptr.add(3 * prev_len + j).cast::<f64>()),
                );
                let (o0, o1, o2, o3, o4) = dft5_f64::<INVERSE>(a0, a1, a2, a3, a4);
                _mm256_storeu_pd(dst_ptr.add(dst_base + j).cast::<f64>(), o0);
                _mm256_storeu_pd(dst_ptr.add(dst_base + j + prev_len).cast::<f64>(), o1);
                _mm256_storeu_pd(dst_ptr.add(dst_base + j + 2 * prev_len).cast::<f64>(), o2);
                _mm256_storeu_pd(dst_ptr.add(dst_base + j + 3 * prev_len).cast::<f64>(), o3);
                _mm256_storeu_pd(dst_ptr.add(dst_base + j + 4 * prev_len).cast::<f64>(), o4);
                j += 2;
            }
            while j < prev_len {
                let mul = |v: Complex<f64>, t: Complex<f64>| {
                    Complex::new(v.re * t.re - v.im * t.im, v.re * t.im + v.im * t.re)
                };
                let a0 = *src_ptr.add(src_base + j);
                let a1 = mul(*src_ptr.add(stride + src_base + j), *tw_ptr.add(j));
                let a2 = mul(
                    *src_ptr.add(2 * stride + src_base + j),
                    *tw_ptr.add(prev_len + j),
                );
                let a3 = mul(
                    *src_ptr.add(3 * stride + src_base + j),
                    *tw_ptr.add(2 * prev_len + j),
                );
                let a4 = mul(
                    *src_ptr.add(4 * stride + src_base + j),
                    *tw_ptr.add(3 * prev_len + j),
                );
                scalar_dft5::<INVERSE>(a0, a1, a2, a3, a4, dst_ptr.add(dst_base + j), prev_len);
                j += 1;
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f64(dst_ptr, pw.as_ptr(), dst.len());
    }
}

// ── f32 radix-5 ─────────────────────────────────────────────────────────────

/// Scalar radix-5 butterfly for f32 (mirrors `scalar_dft5`).
#[inline]
unsafe fn scalar_dft5_f32<const INVERSE: bool>(
    a0: Complex<f32>,
    a1: Complex<f32>,
    a2: Complex<f32>,
    a3: Complex<f32>,
    a4: Complex<f32>,
    out: *mut Complex<f32>,
    prev_len: usize,
) {
    const C1: f32 = 0.309_017_f32;
    const C2: f32 = -0.809_017_f32;
    const S1: f32 = 0.951_056_5_f32;
    const S2: f32 = 0.587_785_25_f32;
    let (s1, s2) = if INVERSE { (S1, S2) } else { (-S1, -S2) };
    let t1_re = a1.re + a4.re;
    let t1_im = a1.im + a4.im;
    let t2_re = a1.re - a4.re;
    let t2_im = a1.im - a4.im;
    let t3_re = a2.re + a3.re;
    let t3_im = a2.im + a3.im;
    let t4_re = a2.re - a3.re;
    let t4_im = a2.im - a3.im;
    let m1_re = t1_re * C1 + t3_re * C2;
    let m1_im = t1_im * C1 + t3_im * C2;
    let m2_re = t1_re * C2 + t3_re * C1;
    let m2_im = t1_im * C2 + t3_im * C1;
    let q3_re = t2_re * s1 + t4_re * s2;
    let q3_im = t2_im * s1 + t4_im * s2;
    let q4_re = t2_re * s2 - t4_re * s1;
    let q4_im = t2_im * s2 - t4_im * s1;
    let a1c_re = a0.re + m1_re;
    let a1c_im = a0.im + m1_im;
    let a2c_re = a0.re + m2_re;
    let a2c_im = a0.im + m2_im;
    *out.add(0) = Complex::new(a0.re + t1_re + t3_re, a0.im + t1_im + t3_im);
    *out.add(prev_len) = Complex::new(a1c_re - q3_im, a1c_im + q3_re);
    *out.add(2 * prev_len) = Complex::new(a2c_re - q4_im, a2c_im + q4_re);
    *out.add(3 * prev_len) = Complex::new(a2c_re + q4_im, a2c_im - q4_re);
    *out.add(4 * prev_len) = Complex::new(a1c_re + q3_im, a1c_im - q3_re);
}

/// AVX2 radix-5 butterfly for 4 Complex<f32> columns simultaneously.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft5_f32<const INVERSE: bool>(
    a0: __m256,
    a1: __m256,
    a2: __m256,
    a3: __m256,
    a4: __m256,
) -> (__m256, __m256, __m256, __m256, __m256) {
    let c1 = _mm256_set1_ps(0.309_017_f32);
    let c2 = _mm256_set1_ps(-0.809_017_f32);
    let (s1, s2) = if INVERSE {
        (
            _mm256_set1_ps(0.951_056_5_f32),
            _mm256_set1_ps(0.587_785_25_f32),
        )
    } else {
        (
            _mm256_set1_ps(-0.951_056_5_f32),
            _mm256_set1_ps(-0.587_785_25_f32),
        )
    };
    let t1 = _mm256_add_ps(a1, a4);
    let t2 = _mm256_sub_ps(a1, a4);
    let t3 = _mm256_add_ps(a2, a3);
    let t4 = _mm256_sub_ps(a2, a3);
    let m1 = _mm256_fmadd_ps(t3, c2, _mm256_mul_ps(t1, c1));
    let m2 = _mm256_fmadd_ps(t3, c1, _mm256_mul_ps(t1, c2));
    let q3 = _mm256_fmadd_ps(t4, s2, _mm256_mul_ps(t2, s1));
    let q4 = _mm256_fmsub_ps(t2, s2, _mm256_mul_ps(t4, s1));
    let a1c = _mm256_add_ps(a0, m1);
    let a2c = _mm256_add_ps(a0, m2);
    let iq3 = rot_pos_i_f32(q3);
    let iq4 = rot_pos_i_f32(q4);
    (
        _mm256_add_ps(a0, _mm256_add_ps(t1, t3)),
        _mm256_add_ps(a1c, iq3),
        _mm256_add_ps(a2c, iq4),
        _mm256_sub_ps(a2c, iq4),
        _mm256_sub_ps(a1c, iq3),
    )
}

/// Flat radix-5 Stockham pass in AVX2+FMA for f32. Processes all `g_count` groups.
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r5_f32<
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
    let stride = g_count * prev_len; // n/5
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        for g in 0..g_count {
            let a0 = *src_ptr.add(g);
            let a1 = *src_ptr.add(stride + g);
            let a2 = *src_ptr.add(2 * stride + g);
            let a3 = *src_ptr.add(3 * stride + g);
            let a4 = *src_ptr.add(4 * stride + g);
            scalar_dft5_f32::<INVERSE>(a0, a1, a2, a3, a4, dst_ptr.add(g * 5), 1);
        }
    } else {
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;
            while j + 3 < prev_len {
                let a0 = _mm256_loadu_ps(src_ptr.add(src_base + j).cast::<f32>());
                let a1r = _mm256_loadu_ps(src_ptr.add(stride + src_base + j).cast::<f32>());
                let a2r = _mm256_loadu_ps(src_ptr.add(2 * stride + src_base + j).cast::<f32>());
                let a3r = _mm256_loadu_ps(src_ptr.add(3 * stride + src_base + j).cast::<f32>());
                let a4r = _mm256_loadu_ps(src_ptr.add(4 * stride + src_base + j).cast::<f32>());
                let a1 = cmul_f32(a1r, _mm256_loadu_ps(tw_ptr.add(j).cast::<f32>()));
                let a2 = cmul_f32(a2r, _mm256_loadu_ps(tw_ptr.add(prev_len + j).cast::<f32>()));
                let a3 = cmul_f32(
                    a3r,
                    _mm256_loadu_ps(tw_ptr.add(2 * prev_len + j).cast::<f32>()),
                );
                let a4 = cmul_f32(
                    a4r,
                    _mm256_loadu_ps(tw_ptr.add(3 * prev_len + j).cast::<f32>()),
                );
                let (o0, o1, o2, o3, o4) = dft5_f32::<INVERSE>(a0, a1, a2, a3, a4);
                _mm256_storeu_ps(dst_ptr.add(dst_base + j).cast::<f32>(), o0);
                _mm256_storeu_ps(dst_ptr.add(dst_base + j + prev_len).cast::<f32>(), o1);
                _mm256_storeu_ps(dst_ptr.add(dst_base + j + 2 * prev_len).cast::<f32>(), o2);
                _mm256_storeu_ps(dst_ptr.add(dst_base + j + 3 * prev_len).cast::<f32>(), o3);
                _mm256_storeu_ps(dst_ptr.add(dst_base + j + 4 * prev_len).cast::<f32>(), o4);
                j += 4;
            }
            while j < prev_len {
                let mul = |v: Complex<f32>, t: Complex<f32>| {
                    Complex::new(v.re * t.re - v.im * t.im, v.re * t.im + v.im * t.re)
                };
                let a0 = *src_ptr.add(src_base + j);
                let a1 = mul(*src_ptr.add(stride + src_base + j), *tw_ptr.add(j));
                let a2 = mul(
                    *src_ptr.add(2 * stride + src_base + j),
                    *tw_ptr.add(prev_len + j),
                );
                let a3 = mul(
                    *src_ptr.add(3 * stride + src_base + j),
                    *tw_ptr.add(2 * prev_len + j),
                );
                let a4 = mul(
                    *src_ptr.add(4 * stride + src_base + j),
                    *tw_ptr.add(3 * prev_len + j),
                );
                scalar_dft5_f32::<INVERSE>(a0, a1, a2, a3, a4, dst_ptr.add(dst_base + j), prev_len);
                j += 1;
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f32(dst_ptr, pw.as_ptr(), dst.len());
    }
}
