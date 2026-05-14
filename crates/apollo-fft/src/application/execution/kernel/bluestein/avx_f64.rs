//! x86_64 AVX/FMA intrinsic kernels for `Complex64` Bluestein pointwise operations.
//!
//! All functions are `unsafe` and require AVX+FMA CPU features.  They are called
//! exclusively through the `BluesteinScalar` trait dispatch in `pointwise.rs`.

use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_64_avx_from_input(
    dst: &mut [Complex64],
    input: &[Complex64],
    twiddle: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_storeu_pd,
        _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), twiddle.len());
    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let in_f = input.as_ptr().cast::<f64>();
    let tw_f = twiddle.as_ptr().cast::<f64>();
    let batches = count / 2;
    for b in 0..batches {
        let f = b * 4;
        let x = _mm256_loadu_pd(in_f.add(f));
        let w = _mm256_loadu_pd(tw_f.add(f));
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_unpackhi_pd(w, w);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(f), yw);
    }
    let tail = batches * 2;
    for i in tail..count {
        dst[i] = input[i] * twiddle[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_64_avx_from_input_conj(
    dst: &mut [Complex64],
    input: &[Complex64],
    twiddle: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd,
        _mm256_storeu_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), twiddle.len());
    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let in_f = input.as_ptr().cast::<f64>();
    let tw_f = twiddle.as_ptr().cast::<f64>();
    let batches = count / 2;
    let neg = _mm256_set1_pd(-1.0);
    for b in 0..batches {
        let f = b * 4;
        let x = _mm256_loadu_pd(in_f.add(f));
        let w = _mm256_loadu_pd(tw_f.add(f));
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_mul_pd(_mm256_unpackhi_pd(w, w), neg);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(f), yw);
    }
    let tail = batches * 2;
    for i in tail..count {
        let in_v = input[i];
        let factor = twiddle[i];
        let re = in_v.re * factor.re + in_v.im * factor.im;
        let im = in_v.im * factor.re - in_v.re * factor.im;
        dst[i] = Complex64::new(re, im);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_64_avx_inplace(
    dst: &mut [Complex64],
    twiddle: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_storeu_pd,
        _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    debug_assert_eq!(dst.len(), twiddle.len());
    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let tw_f = twiddle.as_ptr().cast::<f64>();
    let batches = count / 2;
    for b in 0..batches {
        let f = b * 4;
        let x = _mm256_loadu_pd(dst_f.add(f));
        let w = _mm256_loadu_pd(tw_f.add(f));
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_unpackhi_pd(w, w);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(f), yw);
    }
    let tail = batches * 2;
    for i in tail..count {
        dst[i] *= twiddle[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_64_avx_inplace_inverse(
    dst: &mut [Complex64],
    twiddle: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd,
        _mm256_setr_pd, _mm256_storeu_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    debug_assert_eq!(dst.len(), twiddle.len());
    let len = dst.len();
    if len == 0 {
        return;
    }
    let factor = twiddle[0];
    let re = dst[0].re * factor.re + dst[0].im * factor.im;
    let im = dst[0].im * factor.re - dst[0].re * factor.im;
    dst[0] = Complex64::new(re, im);
    let count = len - 1;
    if count == 0 {
        return;
    }
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let batches = count / 2;
    let neg = _mm256_set1_pd(-1.0);
    for b in 0..batches {
        let dst_offset = 2 + b * 4;
        let x = _mm256_loadu_pd(dst_f.add(dst_offset));
        let first = twiddle[len - 1 - 2 * b];
        let second = twiddle[len - 2 - 2 * b];
        let w = _mm256_setr_pd(first.re, first.im, second.re, second.im);
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_mul_pd(_mm256_unpackhi_pd(w, w), neg);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(dst_offset), yw);
    }
    if count % 2 == 1 {
        let out = &mut dst[len - 1];
        let factor = twiddle[1];
        let re = out.re * factor.re + out.im * factor.im;
        let im = out.im * factor.re - out.re * factor.im;
        *out = Complex64::new(re, im);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn mul_complex_pointwise_64_avx_inplace_inverse_chunk(
    dst: &mut [Complex64],
    twiddle: &[Complex64],
    factor_base: usize,
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd,
        _mm256_setr_pd, _mm256_storeu_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    let count = dst.len();
    if count == 0 {
        return;
    }
    debug_assert!(factor_base < twiddle.len());
    debug_assert!(factor_base + 1 >= count);
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let neg = _mm256_set1_pd(-1.0);
    for b in 0..(count / 2) {
        let dst_offset = b * 4;
        let first = twiddle[factor_base - 2 * b];
        let second = twiddle[factor_base - (2 * b + 1)];
        let w = _mm256_setr_pd(first.re, first.im, second.re, second.im);
        let x = _mm256_loadu_pd(dst_f.add(dst_offset));
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_mul_pd(_mm256_unpackhi_pd(w, w), neg);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(dst_offset), yw);
    }
    if count % 2 == 1 {
        let out = &mut dst[count - 1];
        let factor = twiddle[factor_base - (count - 1)];
        let re = out.re * factor.re + out.im * factor.im;
        let im = out.im * factor.re - out.re * factor.im;
        *out = Complex64::new(re, im);
    }
}

// ── Non-x86 stubs (needed to satisfy the trait impl on other architectures) ──

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_64_avx_from_input(
    _dst: &mut [Complex64],
    _input: &[Complex64],
    _twiddle: &[Complex64],
) {
    unreachable!("AVX not available on this architecture")
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_64_avx_from_input_conj(
    _dst: &mut [Complex64],
    _input: &[Complex64],
    _twiddle: &[Complex64],
) {
    unreachable!()
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_64_avx_inplace(
    _dst: &mut [Complex64],
    _twiddle: &[Complex64],
) {
    unreachable!()
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_64_avx_inplace_inverse(
    _dst: &mut [Complex64],
    _twiddle: &[Complex64],
) {
    unreachable!()
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn mul_complex_pointwise_64_avx_inplace_inverse_chunk(
    _dst: &mut [Complex64],
    _twiddle: &[Complex64],
    _factor_base: usize,
) {
    unreachable!()
}
