use num_complex::{Complex32, Complex64};
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn pointwise_mul_c64_fma(a: &mut [Complex64], b: &[Complex64]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let a_ptr = a.as_mut_ptr() as *mut f64;
    let b_ptr = b.as_ptr() as *const f64;
    let batches = n / 2;
    for i in 0..batches {
        let off = i * 4;
        let av = _mm256_loadu_pd(a_ptr.add(off));
        let bv = _mm256_loadu_pd(b_ptr.add(off));
        let a_re = _mm256_unpacklo_pd(av, av);
        let a_im = _mm256_unpackhi_pd(av, av);
        let b_sw = _mm256_permute_pd(bv, 0b0101);
        let prod = _mm256_mul_pd(a_im, b_sw);
        let res = _mm256_fmaddsub_pd(a_re, bv, prod);
        _mm256_storeu_pd(a_ptr.add(off), res);
    }
    for i in batches * 2..n {
        let av = *a_ptr.add(i * 2);
        let ai = *a_ptr.add(i * 2 + 1);
        let bv = *b_ptr.add(i * 2);
        let bi = *b_ptr.add(i * 2 + 1);
        *a_ptr.add(i * 2) = av * bv - ai * bi;
        *a_ptr.add(i * 2 + 1) = av * bi + ai * bv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn pointwise_mul_c32_fma(a: &mut [Complex32], b: &[Complex32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let a_ptr = a.as_mut_ptr() as *mut f32;
    let b_ptr = b.as_ptr() as *const f32;
    let batches = n / 4;
    for i in 0..batches {
        let off = i * 8;
        let av = _mm256_loadu_ps(a_ptr.add(off));
        let bv = _mm256_loadu_ps(b_ptr.add(off));
        let a_re = _mm256_moveldup_ps(av);
        let a_im = _mm256_movehdup_ps(av);
        let b_sw = _mm256_permute_ps(bv, 0b1011_0001);
        let prod = _mm256_mul_ps(a_im, b_sw);
        let res = _mm256_fmaddsub_ps(a_re, bv, prod);
        _mm256_storeu_ps(a_ptr.add(off), res);
    }
    for i in batches * 4..n {
        let av = *a_ptr.add(i * 2);
        let ai = *a_ptr.add(i * 2 + 1);
        let bv = *b_ptr.add(i * 2);
        let bi = *b_ptr.add(i * 2 + 1);
        *a_ptr.add(i * 2) = av * bv - ai * bi;
        *a_ptr.add(i * 2 + 1) = av * bi + ai * bv;
    }
}

#[inline]
pub(super) fn pointwise_mul_c64(a: &mut [Complex64], b: &[Complex64]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        static HAS_FMA: OnceLock<bool> = OnceLock::new();
        if *HAS_FMA.get_or_init(|| {
            std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
        }) {
            // SAFETY: FMA+AVX confirmed at runtime.
            unsafe { return pointwise_mul_c64_fma(a, b); }
        }
    }
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x *= *y;
    }
}

#[inline]
pub(super) fn pointwise_mul_c32(a: &mut [Complex32], b: &[Complex32]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        static HAS_FMA: OnceLock<bool> = OnceLock::new();
        if *HAS_FMA.get_or_init(|| {
            std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
        }) {
            // SAFETY: FMA+AVX confirmed at runtime.
            unsafe { return pointwise_mul_c32_fma(a, b); }
        }
    }
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x *= *y;
    }
}
