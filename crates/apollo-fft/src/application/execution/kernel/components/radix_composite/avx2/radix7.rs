//! AVX2+FMA flat radix-7 Stockham pass for f64 and f32.
//!
//! Includes DFT-7 constants, scalar butterflies, and AVX2 vectorized butterflies.

use super::helpers::{
    apply_pointwise_f32, apply_pointwise_f64, cmul, cmul_f32, rot_neg_i, rot_neg_i_f32, rot_pos_i,
    rot_pos_i_f32,
};
use num_complex::Complex;
use std::arch::x86_64::{
    __m256, __m256d, _mm256_add_pd, _mm256_add_ps, _mm256_fmadd_pd, _mm256_fmadd_ps,
    _mm256_fnmadd_pd, _mm256_fnmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_mul_pd,
    _mm256_mul_ps, _mm256_set1_pd, _mm256_set1_ps, _mm256_setzero_pd, _mm256_setzero_ps,
    _mm256_storeu_pd, _mm256_storeu_ps, _mm256_sub_pd, _mm256_sub_ps,
};

// ── f64 radix-7 ─────────────────────────────────────────────────────────────

/// Radix-7 DFT constants: cos/sin of 2pi*k/7, k=1,2,3.
const DFT7_C1: f64 = 0.623_489_801_858_733_6;
const DFT7_C2: f64 = -0.222_520_933_956_314_4;
const DFT7_C3: f64 = -0.900_968_867_902_419_1;
const DFT7_S1: f64 = 0.781_831_482_468_029_8;
const DFT7_S2: f64 = 0.974_927_912_181_823_6;
const DFT7_S3: f64 = 0.433_883_739_117_558_2;

/// Scalar radix-7 butterfly mirroring `winograd::radix::dft7_values`.
#[inline]
unsafe fn scalar_dft7<const INVERSE: bool>(
    a: [Complex<f64>; 7],
    out: *mut Complex<f64>,
    prev_len: usize,
) {
    let xr1 = Complex::new(a[1].re + a[6].re, a[1].im + a[6].im);
    let xr2 = Complex::new(a[2].re + a[5].re, a[2].im + a[5].im);
    let xr3 = Complex::new(a[3].re + a[4].re, a[3].im + a[4].im);
    let xi1 = Complex::new(a[1].re - a[6].re, a[1].im - a[6].im);
    let xi2 = Complex::new(a[2].re - a[5].re, a[2].im - a[5].im);
    let xi3 = Complex::new(a[3].re - a[4].re, a[3].im - a[4].im);
    let sixi = |xi: Complex<f64>| {
        if INVERSE {
            Complex::new(-xi.im, xi.re)
        } else {
            Complex::new(xi.im, -xi.re)
        }
    };
    let s1 = sixi(xi1);
    let s2 = sixi(xi2);
    let s3 = sixi(xi3);
    let x0 = a[0];
    let re1 = Complex::new(
        x0.re + xr1.re * DFT7_C1 + xr2.re * DFT7_C2 + xr3.re * DFT7_C3,
        x0.im + xr1.im * DFT7_C1 + xr2.im * DFT7_C2 + xr3.im * DFT7_C3,
    );
    let re2 = Complex::new(
        x0.re + xr1.re * DFT7_C2 + xr2.re * DFT7_C3 + xr3.re * DFT7_C1,
        x0.im + xr1.im * DFT7_C2 + xr2.im * DFT7_C3 + xr3.im * DFT7_C1,
    );
    let re3 = Complex::new(
        x0.re + xr1.re * DFT7_C3 + xr2.re * DFT7_C1 + xr3.re * DFT7_C2,
        x0.im + xr1.im * DFT7_C3 + xr2.im * DFT7_C1 + xr3.im * DFT7_C2,
    );
    let d1 = Complex::new(
        s1.re * DFT7_S1 + s2.re * DFT7_S2 + s3.re * DFT7_S3,
        s1.im * DFT7_S1 + s2.im * DFT7_S2 + s3.im * DFT7_S3,
    );
    let d2 = Complex::new(
        s1.re * DFT7_S2 - s2.re * DFT7_S3 - s3.re * DFT7_S1,
        s1.im * DFT7_S2 - s2.im * DFT7_S3 - s3.im * DFT7_S1,
    );
    let d3 = Complex::new(
        s1.re * DFT7_S3 - s2.re * DFT7_S1 + s3.re * DFT7_S2,
        s1.im * DFT7_S3 - s2.im * DFT7_S1 + s3.im * DFT7_S2,
    );
    *out.add(0) = Complex::new(
        x0.re + xr1.re + xr2.re + xr3.re,
        x0.im + xr1.im + xr2.im + xr3.im,
    );
    *out.add(prev_len) = Complex::new(re1.re + d1.re, re1.im + d1.im);
    *out.add(2 * prev_len) = Complex::new(re2.re + d2.re, re2.im + d2.im);
    *out.add(3 * prev_len) = Complex::new(re3.re + d3.re, re3.im + d3.im);
    *out.add(4 * prev_len) = Complex::new(re3.re - d3.re, re3.im - d3.im);
    *out.add(5 * prev_len) = Complex::new(re2.re - d2.re, re2.im - d2.im);
    *out.add(6 * prev_len) = Complex::new(re1.re - d1.re, re1.im - d1.im);
}

/// AVX2 radix-7 butterfly for 2 complex-f64 columns simultaneously.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft7_f64<const INVERSE: bool>(a: [__m256d; 7]) -> [__m256d; 7] {
    let xr1 = _mm256_add_pd(a[1], a[6]);
    let xr2 = _mm256_add_pd(a[2], a[5]);
    let xr3 = _mm256_add_pd(a[3], a[4]);
    let xi1 = _mm256_sub_pd(a[1], a[6]);
    let xi2 = _mm256_sub_pd(a[2], a[5]);
    let xi3 = _mm256_sub_pd(a[3], a[4]);
    let (s1v, s2v, s3v) = if INVERSE {
        (rot_pos_i(xi1), rot_pos_i(xi2), rot_pos_i(xi3))
    } else {
        (rot_neg_i(xi1), rot_neg_i(xi2), rot_neg_i(xi3))
    };
    let c1 = _mm256_set1_pd(DFT7_C1);
    let c2 = _mm256_set1_pd(DFT7_C2);
    let c3 = _mm256_set1_pd(DFT7_C3);
    let s1 = _mm256_set1_pd(DFT7_S1);
    let s2 = _mm256_set1_pd(DFT7_S2);
    let s3 = _mm256_set1_pd(DFT7_S3);
    let x0 = a[0];
    let re1 = _mm256_fmadd_pd(
        xr3,
        c3,
        _mm256_fmadd_pd(xr2, c2, _mm256_fmadd_pd(xr1, c1, x0)),
    );
    let re2 = _mm256_fmadd_pd(
        xr3,
        c1,
        _mm256_fmadd_pd(xr2, c3, _mm256_fmadd_pd(xr1, c2, x0)),
    );
    let re3 = _mm256_fmadd_pd(
        xr3,
        c2,
        _mm256_fmadd_pd(xr2, c1, _mm256_fmadd_pd(xr1, c3, x0)),
    );
    let d1 = _mm256_fmadd_pd(s3v, s3, _mm256_fmadd_pd(s2v, s2, _mm256_mul_pd(s1v, s1)));
    let d2 = _mm256_fnmadd_pd(s3v, s1, _mm256_fnmadd_pd(s2v, s3, _mm256_mul_pd(s1v, s2)));
    let d3 = _mm256_fmadd_pd(s3v, s2, _mm256_fnmadd_pd(s2v, s1, _mm256_mul_pd(s1v, s3)));
    [
        _mm256_add_pd(x0, _mm256_add_pd(xr1, _mm256_add_pd(xr2, xr3))),
        _mm256_add_pd(re1, d1),
        _mm256_add_pd(re2, d2),
        _mm256_add_pd(re3, d3),
        _mm256_sub_pd(re3, d3),
        _mm256_sub_pd(re2, d2),
        _mm256_sub_pd(re1, d1),
    ]
}

/// Flat radix-7 Stockham pass in AVX2+FMA for f64.
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r7_f64<
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
    let stride = g_count * prev_len; // n/7
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        for g in 0..g_count {
            let mut a = [Complex::new(0.0, 0.0); 7];
            for (k, slot) in a.iter_mut().enumerate() {
                *slot = *src_ptr.add(k * stride + g);
            }
            scalar_dft7::<INVERSE>(a, dst_ptr.add(g * 7), 1);
        }
    } else {
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;
            while j + 1 < prev_len {
                let mut a = [_mm256_setzero_pd(); 7];
                a[0] = _mm256_loadu_pd(src_ptr.add(src_base + j).cast::<f64>());
                for k in 1..7 {
                    let raw = _mm256_loadu_pd(src_ptr.add(k * stride + src_base + j).cast::<f64>());
                    let tw_k = _mm256_loadu_pd(tw_ptr.add((k - 1) * prev_len + j).cast::<f64>());
                    a[k] = cmul(raw, tw_k);
                }
                let o = dft7_f64::<INVERSE>(a);
                for (k, ok) in o.iter().enumerate() {
                    _mm256_storeu_pd(dst_ptr.add(dst_base + j + k * prev_len).cast::<f64>(), *ok);
                }
                j += 2;
            }
            while j < prev_len {
                let mul = |v: Complex<f64>, t: Complex<f64>| {
                    Complex::new(v.re * t.re - v.im * t.im, v.re * t.im + v.im * t.re)
                };
                let mut a = [Complex::new(0.0, 0.0); 7];
                a[0] = *src_ptr.add(src_base + j);
                for k in 1..7 {
                    a[k] = mul(
                        *src_ptr.add(k * stride + src_base + j),
                        *tw_ptr.add((k - 1) * prev_len + j),
                    );
                }
                scalar_dft7::<INVERSE>(a, dst_ptr.add(dst_base + j), prev_len);
                j += 1;
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f64(dst_ptr, pw.as_ptr(), dst.len());
    }
}

// ── f32 radix-7 ─────────────────────────────────────────────────────────────

/// Scalar radix-7 butterfly for f32 (mirrors `scalar_dft7`).
#[inline]
unsafe fn scalar_dft7_f32<const INVERSE: bool>(
    a: [Complex<f32>; 7],
    out: *mut Complex<f32>,
    prev_len: usize,
) {
    const C1: f32 = 0.623_489_8_f32;
    const C2: f32 = -0.222_520_93_f32;
    const C3: f32 = -0.900_968_87_f32;
    const S1: f32 = 0.781_831_5_f32;
    const S2: f32 = 0.974_927_9_f32;
    const S3: f32 = 0.433_883_74_f32;
    let xr1 = Complex::new(a[1].re + a[6].re, a[1].im + a[6].im);
    let xr2 = Complex::new(a[2].re + a[5].re, a[2].im + a[5].im);
    let xr3 = Complex::new(a[3].re + a[4].re, a[3].im + a[4].im);
    let xi1 = Complex::new(a[1].re - a[6].re, a[1].im - a[6].im);
    let xi2 = Complex::new(a[2].re - a[5].re, a[2].im - a[5].im);
    let xi3 = Complex::new(a[3].re - a[4].re, a[3].im - a[4].im);
    let sixi = |xi: Complex<f32>| {
        if INVERSE {
            Complex::new(-xi.im, xi.re)
        } else {
            Complex::new(xi.im, -xi.re)
        }
    };
    let s1 = sixi(xi1);
    let s2 = sixi(xi2);
    let s3 = sixi(xi3);
    let x0 = a[0];
    let re1 = Complex::new(
        x0.re + xr1.re * C1 + xr2.re * C2 + xr3.re * C3,
        x0.im + xr1.im * C1 + xr2.im * C2 + xr3.im * C3,
    );
    let re2 = Complex::new(
        x0.re + xr1.re * C2 + xr2.re * C3 + xr3.re * C1,
        x0.im + xr1.im * C2 + xr2.im * C3 + xr3.im * C1,
    );
    let re3 = Complex::new(
        x0.re + xr1.re * C3 + xr2.re * C1 + xr3.re * C2,
        x0.im + xr1.im * C3 + xr2.im * C1 + xr3.im * C2,
    );
    let d1 = Complex::new(
        s1.re * S1 + s2.re * S2 + s3.re * S3,
        s1.im * S1 + s2.im * S2 + s3.im * S3,
    );
    let d2 = Complex::new(
        s1.re * S2 - s2.re * S3 - s3.re * S1,
        s1.im * S2 - s2.im * S3 - s3.im * S1,
    );
    let d3 = Complex::new(
        s1.re * S3 - s2.re * S1 + s3.re * S2,
        s1.im * S3 - s2.im * S1 + s3.im * S2,
    );
    *out.add(0) = Complex::new(
        x0.re + xr1.re + xr2.re + xr3.re,
        x0.im + xr1.im + xr2.im + xr3.im,
    );
    *out.add(prev_len) = Complex::new(re1.re + d1.re, re1.im + d1.im);
    *out.add(2 * prev_len) = Complex::new(re2.re + d2.re, re2.im + d2.im);
    *out.add(3 * prev_len) = Complex::new(re3.re + d3.re, re3.im + d3.im);
    *out.add(4 * prev_len) = Complex::new(re3.re - d3.re, re3.im - d3.im);
    *out.add(5 * prev_len) = Complex::new(re2.re - d2.re, re2.im - d2.im);
    *out.add(6 * prev_len) = Complex::new(re1.re - d1.re, re1.im - d1.im);
}

/// AVX2 radix-7 butterfly for 4 Complex<f32> columns simultaneously.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dft7_f32<const INVERSE: bool>(a: [__m256; 7]) -> [__m256; 7] {
    let xr1 = _mm256_add_ps(a[1], a[6]);
    let xr2 = _mm256_add_ps(a[2], a[5]);
    let xr3 = _mm256_add_ps(a[3], a[4]);
    let xi1 = _mm256_sub_ps(a[1], a[6]);
    let xi2 = _mm256_sub_ps(a[2], a[5]);
    let xi3 = _mm256_sub_ps(a[3], a[4]);
    let (s1v, s2v, s3v) = if INVERSE {
        (rot_pos_i_f32(xi1), rot_pos_i_f32(xi2), rot_pos_i_f32(xi3))
    } else {
        (rot_neg_i_f32(xi1), rot_neg_i_f32(xi2), rot_neg_i_f32(xi3))
    };
    let c1 = _mm256_set1_ps(0.623_489_8_f32);
    let c2 = _mm256_set1_ps(-0.222_520_93_f32);
    let c3 = _mm256_set1_ps(-0.900_968_87_f32);
    let s1 = _mm256_set1_ps(0.781_831_5_f32);
    let s2 = _mm256_set1_ps(0.974_927_9_f32);
    let s3 = _mm256_set1_ps(0.433_883_74_f32);
    let x0 = a[0];
    let re1 = _mm256_fmadd_ps(
        xr3,
        c3,
        _mm256_fmadd_ps(xr2, c2, _mm256_fmadd_ps(xr1, c1, x0)),
    );
    let re2 = _mm256_fmadd_ps(
        xr3,
        c1,
        _mm256_fmadd_ps(xr2, c3, _mm256_fmadd_ps(xr1, c2, x0)),
    );
    let re3 = _mm256_fmadd_ps(
        xr3,
        c2,
        _mm256_fmadd_ps(xr2, c1, _mm256_fmadd_ps(xr1, c3, x0)),
    );
    let d1 = _mm256_fmadd_ps(s3v, s3, _mm256_fmadd_ps(s2v, s2, _mm256_mul_ps(s1v, s1)));
    let d2 = _mm256_fnmadd_ps(s3v, s1, _mm256_fnmadd_ps(s2v, s3, _mm256_mul_ps(s1v, s2)));
    let d3 = _mm256_fmadd_ps(s3v, s2, _mm256_fnmadd_ps(s2v, s1, _mm256_mul_ps(s1v, s3)));
    [
        _mm256_add_ps(x0, _mm256_add_ps(xr1, _mm256_add_ps(xr2, xr3))),
        _mm256_add_ps(re1, d1),
        _mm256_add_ps(re2, d2),
        _mm256_add_ps(re3, d3),
        _mm256_sub_ps(re3, d3),
        _mm256_sub_ps(re2, d2),
        _mm256_sub_ps(re1, d1),
    ]
}

/// Flat radix-7 Stockham pass in AVX2+FMA for f32 (4 complex per __m256).
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r7_f32<
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
    let stride = g_count * prev_len; // n/7
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        for g in 0..g_count {
            let mut a = [Complex::new(0.0_f32, 0.0_f32); 7];
            for (k, slot) in a.iter_mut().enumerate() {
                *slot = *src_ptr.add(k * stride + g);
            }
            scalar_dft7_f32::<INVERSE>(a, dst_ptr.add(g * 7), 1);
        }
    } else {
        let tw_ptr = tw.as_ptr();
        for g in 0..g_count {
            let src_base = g * prev_len;
            let dst_base = g * stage_chunk;
            let mut j = 0usize;
            while j + 3 < prev_len {
                let mut a = [_mm256_setzero_ps(); 7];
                a[0] = _mm256_loadu_ps(src_ptr.add(src_base + j).cast::<f32>());
                for k in 1..7 {
                    let raw = _mm256_loadu_ps(src_ptr.add(k * stride + src_base + j).cast::<f32>());
                    let tw_k = _mm256_loadu_ps(tw_ptr.add((k - 1) * prev_len + j).cast::<f32>());
                    a[k] = cmul_f32(raw, tw_k);
                }
                let o = dft7_f32::<INVERSE>(a);
                for (k, ok) in o.iter().enumerate() {
                    _mm256_storeu_ps(dst_ptr.add(dst_base + j + k * prev_len).cast::<f32>(), *ok);
                }
                j += 4;
            }
            while j < prev_len {
                let mul = |v: Complex<f32>, t: Complex<f32>| {
                    Complex::new(v.re * t.re - v.im * t.im, v.re * t.im + v.im * t.re)
                };
                let mut a = [Complex::new(0.0_f32, 0.0_f32); 7];
                a[0] = *src_ptr.add(src_base + j);
                for k in 1..7 {
                    a[k] = mul(
                        *src_ptr.add(k * stride + src_base + j),
                        *tw_ptr.add((k - 1) * prev_len + j),
                    );
                }
                scalar_dft7_f32::<INVERSE>(a, dst_ptr.add(dst_base + j), prev_len);
                j += 1;
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f32(dst_ptr, pw.as_ptr(), dst.len());
    }
}
