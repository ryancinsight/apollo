//! x86_64 AVX/FMA intrinsic kernels for `Complex32` Bluestein pointwise operations.
//!
//! All functions are `unsafe` and require AVX+FMA CPU features.  They are called
//! exclusively through the `BluesteinScalar` trait dispatch in `pointwise.rs`.

use num_complex::Complex32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_32_avx_from_input(
    dst: &mut [Complex32],
    input: &[Complex32],
    twiddle: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_storeu_ps,
    };
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), twiddle.len());
    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let in_f = input.as_ptr().cast::<f32>();
    let tw_f = twiddle.as_ptr().cast::<f32>();
    let batches = count / 4;
    for b in 0..batches {
        let f = b * 8;
        let x = _mm256_loadu_ps(in_f.add(f));
        let w = _mm256_loadu_ps(tw_f.add(f));
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_movehdup_ps(w);
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(f), yw);
    }
    let tail = batches * 4;
    for i in tail..count {
        dst[i] = input[i] * twiddle[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_32_avx_from_input_conj(
    dst: &mut [Complex32],
    input: &[Complex32],
    twiddle: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), twiddle.len());
    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let in_f = input.as_ptr().cast::<f32>();
    let tw_f = twiddle.as_ptr().cast::<f32>();
    let batches = count / 4;
    let neg = _mm256_set1_ps(-1.0);
    for b in 0..batches {
        let f = b * 8;
        let x = _mm256_loadu_ps(in_f.add(f));
        let w = _mm256_loadu_ps(tw_f.add(f));
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_mul_ps(_mm256_movehdup_ps(w), neg);
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(f), yw);
    }
    let tail = batches * 4;
    for i in tail..count {
        let in_v = input[i];
        let factor = twiddle[i];
        let re = in_v.re * factor.re + in_v.im * factor.im;
        let im = in_v.im * factor.re - in_v.re * factor.im;
        dst[i] = Complex32::new(re, im);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_32_avx_inplace(
    dst: &mut [Complex32],
    twiddle: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_storeu_ps,
    };
    debug_assert_eq!(dst.len(), twiddle.len());
    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let tw_f = twiddle.as_ptr().cast::<f32>();
    let batches = count / 4;
    for b in 0..batches {
        let f = b * 8;
        let x = _mm256_loadu_ps(dst_f.add(f));
        let w = _mm256_loadu_ps(tw_f.add(f));
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_movehdup_ps(w);
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(f), yw);
    }
    let tail = batches * 4;
    for i in tail..count {
        dst[i] *= twiddle[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_32_avx_inplace_inverse(
    dst: &mut [Complex32],
    twiddle: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_set1_ps, _mm256_setr_ps, _mm256_storeu_ps,
    };
    debug_assert_eq!(dst.len(), twiddle.len());
    let len = dst.len();
    if len == 0 {
        return;
    }
    let factor = twiddle[0];
    let re = dst[0].re * factor.re + dst[0].im * factor.im;
    let im = dst[0].im * factor.re - dst[0].re * factor.im;
    dst[0] = Complex32::new(re, im);
    let count = len - 1;
    if count == 0 {
        return;
    }
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let batches = count / 4;
    let neg = _mm256_set1_ps(-1.0);
    for b in 0..batches {
        let dst_offset = 2 + b * 8;
        let x = _mm256_loadu_ps(dst_f.add(dst_offset));
        let i0 = len - 1 - 4 * b;
        let i1 = len - 2 - 4 * b;
        let i2 = len - 3 - 4 * b;
        let i3 = len - 4 - 4 * b;
        let w0 = twiddle[i0];
        let w1 = twiddle[i1];
        let w2 = twiddle[i2];
        let w3 = twiddle[i3];
        let w = _mm256_setr_ps(w0.re, w0.im, w1.re, w1.im, w2.re, w2.im, w3.re, w3.im);
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_mul_ps(_mm256_movehdup_ps(w), neg);
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(dst_offset), yw);
    }
    let rem = count % 4;
    if rem != 0 {
        let base = (count / 4) * 4 + 1;
        let batches = count / 4;
        for k in 0..rem {
            let out = &mut dst[base + k];
            let factor = twiddle[len - (4 * batches + k + 1)];
            let re = out.re * factor.re + out.im * factor.im;
            let im = out.im * factor.re - out.re * factor.im;
            *out = Complex32::new(re, im);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_32_avx_inplace_inverse_chunk(
    dst: &mut [Complex32],
    twiddle: &[Complex32],
    factor_base: usize,
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_set1_ps, _mm256_setr_ps, _mm256_storeu_ps,
    };
    let count = dst.len();
    if count == 0 {
        return;
    }
    debug_assert!(factor_base < twiddle.len());
    debug_assert!(factor_base + 1 >= count);
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let neg = _mm256_set1_ps(-1.0);
    for b in 0..(count / 4) {
        let dst_offset = b * 8;
        let w0 = twiddle[factor_base - 4 * b];
        let w1 = twiddle[factor_base - (4 * b + 1)];
        let w2 = twiddle[factor_base - (4 * b + 2)];
        let w3 = twiddle[factor_base - (4 * b + 3)];
        let w = _mm256_setr_ps(w0.re, w0.im, w1.re, w1.im, w2.re, w2.im, w3.re, w3.im);
        let x = _mm256_loadu_ps(dst_f.add(dst_offset));
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_mul_ps(_mm256_movehdup_ps(w), neg);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(dst_offset), yw);
    }
    if !count.is_multiple_of(4) {
        let base = (count / 4) * 4;
        for k in 0..(count % 4) {
            let out = &mut dst[base + k];
            let factor = twiddle[factor_base - (base + k)];
            let re = out.re * factor.re + out.im * factor.im;
            let im = out.im * factor.re - out.re * factor.im;
            *out = Complex32::new(re, im);
        }
    }
}

// ── Non-x86 stubs ────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_32_avx_from_input(
    _dst: &mut [Complex32],
    _input: &[Complex32],
    _twiddle: &[Complex32],
) {
    unreachable!("AVX not available on this architecture")
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_32_avx_from_input_conj(
    _dst: &mut [Complex32],
    _input: &[Complex32],
    _twiddle: &[Complex32],
) {
    unreachable!()
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_32_avx_inplace(
    _dst: &mut [Complex32],
    _twiddle: &[Complex32],
) {
    unreachable!()
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_32_avx_inplace_inverse(
    _dst: &mut [Complex32],
    _twiddle: &[Complex32],
) {
    unreachable!()
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_32_avx_inplace_inverse_chunk(
    _dst: &mut [Complex32],
    _twiddle: &[Complex32],
    _factor_base: usize,
) {
    unreachable!()
}
