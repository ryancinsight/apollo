//! Shared AVX2+FMA primitives: complex multiply, ±i rotation, pointwise apply.
//!
//! These are used by all radix modules (r2-r7) for both f64 and f32.

use eunomia::Complex;
use std::arch::x86_64::{
    __m128, __m256, __m256d, _mm256_fmaddsub_pd, _mm256_fmaddsub_ps, _mm256_loadu_pd,
    _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_pd, _mm256_mul_ps,
    _mm256_permute_pd, _mm256_permute_ps, _mm256_set_pd, _mm256_set_ps, _mm256_storeu_pd,
    _mm256_storeu_ps, _mm256_xor_pd, _mm256_xor_ps, _mm_fmaddsub_ps, _mm_loadu_ps, _mm_movehdup_ps,
    _mm_moveldup_ps, _mm_mul_ps, _mm_permute_ps, _mm_storeu_ps,
};

// ── f64 primitives ──────────────────────────────────────────────────────────

/// Complex multiply: (a_re + i·a_im) × (b_re + i·b_im) for 2 complex-f64 pairs.
///
/// Uses the fmaddsub pattern:
///   a_re = broadcast_re(a), a_im = broadcast_im(a), b_sw = swap_re_im(b)
///   result = fmaddsub(a_re, b, a_im * b_sw)
///          = [a_re·b_re - a_im·b_im, a_re·b_im + a_im·b_re, ...]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn cmul(a: __m256d, b: __m256d) -> __m256d {
    let a_re = _mm256_permute_pd(a, 0b0000); // [re0, re0, re1, re1]
    let a_im = _mm256_permute_pd(a, 0b1111); // [im0, im0, im1, im1]
    let b_sw = _mm256_permute_pd(b, 0b0101); // [im0, re0, im1, re1]
    _mm256_fmaddsub_pd(a_re, b, _mm256_mul_pd(a_im, b_sw))
}

/// Multiply v by -i: (re+i·im) → (im, -re).
/// permute(v, 0b0101) = [im0, re0, im1, re1]; XOR negates re positions (1, 3).
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rot_neg_i(v: __m256d) -> __m256d {
    let sign = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    _mm256_xor_pd(_mm256_permute_pd(v, 0b0101), sign)
}

/// Multiply v by +i: (re+i·im) → (-im, re).
/// permute(v, 0b0101) = [im0, re0, im1, re1]; XOR negates im positions (0, 2).
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rot_pos_i(v: __m256d) -> __m256d {
    let sign = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
    _mm256_xor_pd(_mm256_permute_pd(v, 0b0101), sign)
}

/// Apply pointwise frequency-domain multiply to dst (inline after butterfly).
/// `pw` covers `dst[0..dst.len()]` element-wise. Used on the final stage only.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn apply_pointwise_f64(
    dst: *mut Complex<f64>,
    pw: *const Complex<f64>,
    len: usize,
) {
    let mut i = 0usize;
    while i + 1 < len {
        let d = _mm256_loadu_pd(dst.add(i) as *const f64);
        let p = _mm256_loadu_pd(pw.add(i).cast::<f64>());
        _mm256_storeu_pd(dst.add(i).cast::<f64>(), cmul(d, p));
        i += 2;
    }
    if i < len {
        let d0 = *dst.add(i);
        let p0 = *pw.add(i);
        *dst.add(i) = Complex {
            re: d0.re * p0.re - d0.im * p0.im,
            im: d0.re * p0.im + d0.im * p0.re,
        };
    }
}

// ── f32 primitives ──────────────────────────────────────────────────────────

/// Complex multiply for 4 Complex<f32> pairs simultaneously (8 f32 = __m256).
///
/// Layout: [re0,im0,re1,im1,re2,im2,re3,im3].
/// moveldup/movehdup broadcast re/im; permute(0xB1) swaps re↔im within each pair.
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn cmul_f32(a: __m256, b: __m256) -> __m256 {
    let a_re = _mm256_moveldup_ps(a); // [re0,re0,re1,re1,re2,re2,re3,re3]
    let a_im = _mm256_movehdup_ps(a); // [im0,im0,im1,im1,im2,im2,im3,im3]
    let b_sw = _mm256_permute_ps(b, 0xB1); // swap re↔im: [im0,re0,im1,re1,...]
    _mm256_fmaddsub_ps(a_re, b, _mm256_mul_ps(a_im, b_sw))
}

/// Multiply 4 Complex<f32> by -i: (re+i·im) → (im, -re).
/// permute(0xB1) swaps pairs → [im,re,...]; XOR negates re positions (1,3,5,7).
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rot_neg_i_f32(v: __m256) -> __m256 {
    let sign = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
    _mm256_xor_ps(_mm256_permute_ps(v, 0xB1), sign)
}

/// Multiply 4 Complex<f32> by +i: (re+i·im) → (-im, re).
/// permute(0xB1) swaps pairs → [im,re,...]; XOR negates im positions (0,2,4,6).
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rot_pos_i_f32(v: __m256) -> __m256 {
    let sign = _mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    _mm256_xor_ps(_mm256_permute_ps(v, 0xB1), sign)
}

/// Apply pointwise multiply to f32 dst.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn apply_pointwise_f32(
    dst: *mut Complex<f32>,
    pw: *const Complex<f32>,
    len: usize,
) {
    let mut i = 0usize;
    while i + 3 < len {
        let d = _mm256_loadu_ps(dst.add(i) as *const f32);
        let p = _mm256_loadu_ps(pw.add(i).cast::<f32>());
        _mm256_storeu_ps(dst.add(i).cast::<f32>(), cmul_f32(d, p));
        i += 4;
    }
    // 2-wide tail
    while i + 1 < len {
        let d = _mm_loadu_ps(dst.add(i) as *const f32);
        let p = _mm_loadu_ps(pw.add(i).cast::<f32>());
        let d_re = _mm_moveldup_ps(d);
        let d_im = _mm_movehdup_ps(d);
        let p_sw = _mm_permute_ps(p, 0xB1);
        let result = _mm_fmaddsub_ps(d_re, p, _mm_mul_ps(d_im, p_sw));
        _mm_storeu_ps(dst.add(i).cast::<f32>(), result);
        i += 2;
    }
    if i < len {
        let d = *dst.add(i);
        let p = *pw.add(i);
        *dst.add(i) = Complex {
            re: d.re * p.re - d.im * p.im,
            im: d.re * p.im + d.im * p.re,
        };
    }
}

/// cmul for 2 Complex<f32> via __m128.
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn cmul_f32_128(a: __m128, b: __m128) -> __m128 {
    let a_re = _mm_moveldup_ps(a);
    let a_im = _mm_movehdup_ps(a);
    let b_sw = _mm_permute_ps(b, 0xB1);
    _mm_fmaddsub_ps(a_re, b, _mm_mul_ps(a_im, b_sw))
}
