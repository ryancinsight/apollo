use hermes_simd::{interleaved_complex_mul_assign, PreferredArch};
use num_complex::{Complex32, Complex64};
use std::sync::OnceLock;

#[inline]
fn mul_pair_precise<const CONJ_B: bool>(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    let bi = if CONJ_B { -bi } else { bi };
    (ar * br - ai * bi, ar * bi + ai * br)
}

#[inline]
fn mul_pair_reduced<const CONJ_B: bool>(ar: f32, ai: f32, br: f32, bi: f32) -> (f32, f32) {
    let bi = if CONJ_B { -bi } else { bi };
    (ar * br - ai * bi, ar * bi + ai * br)
}

#[inline]
fn pointwise_mul_precise_hermes<const CONJ_B: bool>(a: &mut [Complex64], b: &[Complex64]) {
    let scalar_len = a.len() * 2;
    // SAFETY: num_complex::Complex64 stores adjacent `re, im` f64 lanes; this
    // file already uses the same representation for the AVX/FMA hot path.
    let a_lanes =
        unsafe { core::slice::from_raw_parts_mut(a.as_mut_ptr().cast::<f64>(), scalar_len) };
    // SAFETY: same representation as `a_lanes`; `b` is immutable and length-matched by caller.
    let b_lanes = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<f64>(), scalar_len) };
    interleaved_complex_mul_assign::<f64, PreferredArch, CONJ_B>(a_lanes, b_lanes)
        .expect("pointwise operands must have matching complex lengths");
}

#[inline]
fn pointwise_mul_reduced_hermes<const CONJ_B: bool>(a: &mut [Complex32], b: &[Complex32]) {
    let scalar_len = a.len() * 2;
    // SAFETY: num_complex::Complex32 stores adjacent `re, im` f32 lanes; this
    // file already uses the same representation for the AVX/FMA hot path.
    let a_lanes =
        unsafe { core::slice::from_raw_parts_mut(a.as_mut_ptr().cast::<f32>(), scalar_len) };
    // SAFETY: same representation as `a_lanes`; `b` is immutable and length-matched by caller.
    let b_lanes = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<f32>(), scalar_len) };
    interleaved_complex_mul_assign::<f32, PreferredArch, CONJ_B>(a_lanes, b_lanes)
        .expect("pointwise operands must have matching complex lengths");
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn pointwise_mul_precise_fma<const CONJ_B: bool>(a: &mut [Complex64], b: &[Complex64]) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set_pd,
        _mm256_setzero_pd, _mm256_storeu_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd, _mm256_xor_pd,
    };
    let n = a.len();
    let a_ptr = a.as_mut_ptr().cast::<f64>();
    let b_ptr = b.as_ptr().cast::<f64>();
    let sign_mask = if CONJ_B {
        _mm256_set_pd(-0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64)
    } else {
        _mm256_setzero_pd()
    };
    let batches = n / 2;
    for i in 0..batches {
        let off = i * 4;
        let av = _mm256_loadu_pd(a_ptr.add(off));
        let bv = _mm256_xor_pd(_mm256_loadu_pd(b_ptr.add(off)), sign_mask);
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
        let (re, im) = mul_pair_precise::<CONJ_B>(av, ai, bv, bi);
        *a_ptr.add(i * 2) = re;
        *a_ptr.add(i * 2 + 1) = im;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn pointwise_mul_reduced_fma<const CONJ_B: bool>(a: &mut [Complex32], b: &[Complex32]) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_set_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_xor_ps,
    };
    let n = a.len();
    let a_ptr = a.as_mut_ptr().cast::<f32>();
    let b_ptr = b.as_ptr().cast::<f32>();
    let sign_mask = if CONJ_B {
        _mm256_set_ps(
            -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32,
        )
    } else {
        _mm256_setzero_ps()
    };
    let batches = n / 4;
    for i in 0..batches {
        let off = i * 8;
        let av = _mm256_loadu_ps(a_ptr.add(off));
        let bv = _mm256_xor_ps(_mm256_loadu_ps(b_ptr.add(off)), sign_mask);
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
        let (re, im) = mul_pair_reduced::<CONJ_B>(av, ai, bv, bi);
        *a_ptr.add(i * 2) = re;
        *a_ptr.add(i * 2 + 1) = im;
    }
}

#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) fn pointwise_mul_precise<
    const CONJ_B: bool,
>(
    a: &mut [Complex64],
    b: &[Complex64],
) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        static HAS_FMA: OnceLock<bool> = OnceLock::new();
        if *HAS_FMA.get_or_init(|| {
            std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
        }) {
            // SAFETY: FMA+AVX confirmed at runtime.
            unsafe {
                return pointwise_mul_precise_fma::<CONJ_B>(a, b);
            }
        }
    }
    pointwise_mul_precise_hermes::<CONJ_B>(a, b);
}

#[inline]
pub(in crate::application::execution::kernel::mixed_radix::scalar) fn pointwise_mul_reduced<
    const CONJ_B: bool,
>(
    a: &mut [Complex32],
    b: &[Complex32],
) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        static HAS_FMA: OnceLock<bool> = OnceLock::new();
        if *HAS_FMA.get_or_init(|| {
            std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
        }) {
            // SAFETY: FMA+AVX confirmed at runtime.
            unsafe {
                return pointwise_mul_reduced_fma::<CONJ_B>(a, b);
            }
        }
    }
    pointwise_mul_reduced_hermes::<CONJ_B>(a, b);
}
