//! AVX2+FMA flat radix-2 Stockham pass for f64 and f32.
//!
//! Vectorizes the trailing radix-2 stage of odd-power-of-two decompositions
//! (e.g. N=32/128/512 lower to `[4,..,4,2]`), previously scalar.
//! Butterfly: `b0 = a0 + tw·a1`, `b1 = a0 − tw·a1`;
//! direction is carried by the precomputed twiddle table.

use super::helpers::{apply_pointwise_f32, apply_pointwise_f64, cmul, cmul_f32};
use eunomia::Complex;
use std::arch::x86_64::{
    _mm256_add_pd, _mm256_add_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_storeu_pd,
    _mm256_storeu_ps, _mm256_sub_pd, _mm256_sub_ps,
};

/// Flat radix-2 Stockham pass in AVX2+FMA for f64.
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r2_f64(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    prev_len: usize,
    g_count: usize,
    stage_chunk: usize,
    tw: &[Complex<f64>],
    pointwise: Option<&[Complex<f64>]>,
) {
    let stride = g_count * prev_len; // n/2
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        for g in 0..g_count {
            let a0 = *src_ptr.add(g);
            let a1 = *src_ptr.add(stride + g);
            let d = dst_ptr.add(g * 2);
            *d.add(0) = Complex::new(a0.re + a1.re, a0.im + a1.im);
            *d.add(1) = Complex::new(a0.re - a1.re, a0.im - a1.im);
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
                let a1 = cmul(a1r, _mm256_loadu_pd(tw_ptr.add(j).cast::<f64>()));
                _mm256_storeu_pd(
                    dst_ptr.add(dst_base + j).cast::<f64>(),
                    _mm256_add_pd(a0, a1),
                );
                _mm256_storeu_pd(
                    dst_ptr.add(dst_base + j + prev_len).cast::<f64>(),
                    _mm256_sub_pd(a0, a1),
                );
                j += 2;
            }
            while j < prev_len {
                let v = *src_ptr.add(stride + src_base + j);
                let t = *tw_ptr.add(j);
                let a1 = Complex::new(v.re * t.re - v.im * t.im, v.re * t.im + v.im * t.re);
                let a0 = *src_ptr.add(src_base + j);
                *dst_ptr.add(dst_base + j) = Complex::new(a0.re + a1.re, a0.im + a1.im);
                *dst_ptr.add(dst_base + j + prev_len) = Complex::new(a0.re - a1.re, a0.im - a1.im);
                j += 1;
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f64(dst_ptr, pw.as_ptr(), dst.len());
    }
}

/// Flat radix-2 Stockham pass in AVX2+FMA for f32 (4 complex per __m256).
#[target_feature(enable = "avx2,fma")]
pub(in crate::application::execution::kernel::components::radix_composite) unsafe fn flat_pass_r2_f32(
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    prev_len: usize,
    g_count: usize,
    stage_chunk: usize,
    tw: &[Complex<f32>],
    pointwise: Option<&[Complex<f32>]>,
) {
    let stride = g_count * prev_len; // n/2
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    if prev_len == 1 {
        for g in 0..g_count {
            let a0 = *src_ptr.add(g);
            let a1 = *src_ptr.add(stride + g);
            let d = dst_ptr.add(g * 2);
            *d.add(0) = Complex::new(a0.re + a1.re, a0.im + a1.im);
            *d.add(1) = Complex::new(a0.re - a1.re, a0.im - a1.im);
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
                let a1 = cmul_f32(a1r, _mm256_loadu_ps(tw_ptr.add(j).cast::<f32>()));
                _mm256_storeu_ps(
                    dst_ptr.add(dst_base + j).cast::<f32>(),
                    _mm256_add_ps(a0, a1),
                );
                _mm256_storeu_ps(
                    dst_ptr.add(dst_base + j + prev_len).cast::<f32>(),
                    _mm256_sub_ps(a0, a1),
                );
                j += 4;
            }
            while j < prev_len {
                let v = *src_ptr.add(stride + src_base + j);
                let t = *tw_ptr.add(j);
                let a1 = Complex::new(v.re * t.re - v.im * t.im, v.re * t.im + v.im * t.re);
                let a0 = *src_ptr.add(src_base + j);
                *dst_ptr.add(dst_base + j) = Complex::new(a0.re + a1.re, a0.im + a1.im);
                *dst_ptr.add(dst_base + j + prev_len) = Complex::new(a0.re - a1.re, a0.im - a1.im);
                j += 1;
            }
        }
    }

    if let Some(pw) = pointwise {
        apply_pointwise_f32(dst_ptr, pw.as_ptr(), dst.len());
    }
}
