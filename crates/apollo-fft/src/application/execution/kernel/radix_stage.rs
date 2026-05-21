//! Shared normalization primitive for radix and Rader kernel modules.
//!
//! ## Contents
//!
//! - `NormalizeSlice`: sealed trait abstracting the AVX-dispatched scale pass.
//! - `normalize_inplace`: SSOT 1/N scale pass, used by all inverse paths.

use num_complex::{Complex32, Complex64};

// ── Private AVX fast paths ─────────────────────────────────────────────────────

/// Scale a `Complex64` slice in-place by `scale` using AVX broadcast-multiply.
///
/// # Safety
///
/// Caller must ensure AVX is available (gated by `is_x86_feature_detected!`).
/// The slice must be non-empty.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn normalize_precise_avx(data: &mut [Complex64], scale: f64) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd};

    let s = _mm256_set1_pd(scale);
    let ptr = data.as_mut_ptr().cast::<f64>();
    let len = data.len();
    // Each YMM register holds 2 Complex64 = 4 f64 lanes.
    let batches = len / 2;
    for b in 0..batches {
        let off = b * 4;
        let x = _mm256_loadu_pd(ptr.add(off));
        _mm256_storeu_pd(ptr.add(off), _mm256_mul_pd(x, s));
    }
    for i in batches * 2..len {
        data[i] *= scale;
    }
}

/// Scale a `Complex32` slice in-place by `scale` using AVX broadcast-multiply.
///
/// # Safety
///
/// Caller must ensure AVX is available (gated by `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn normalize_reduced_avx(data: &mut [Complex32], scale: f32) {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps};

    let s = _mm256_set1_ps(scale);
    let ptr = data.as_mut_ptr().cast::<f32>();
    let len = data.len();
    // Each YMM register holds 4 Complex32 = 8 f32 lanes.
    let batches = len / 4;
    for b in 0..batches {
        let off = b * 8;
        let x = _mm256_loadu_ps(ptr.add(off));
        _mm256_storeu_ps(ptr.add(off), _mm256_mul_ps(x, s));
    }
    for i in batches * 4..len {
        data[i] *= scale;
    }
}

// ── Sealed trait ──────────────────────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
}

/// Sealed normalization trait. Implemented only for `Complex32` and `Complex64`.
///
/// Dispatches to an AVX broadcast-multiply kernel on x86_64 when AVX is
/// available at runtime; falls back to the scalar loop otherwise.
pub(crate) trait NormalizeSlice: sealed::Sealed + Sized {
    type Scale: Copy;
    fn normalize_slice(data: &mut [Self], scale: Self::Scale);
}

impl sealed::Sealed for Complex64 {}
impl NormalizeSlice for Complex64 {
    type Scale = f64;
    #[inline]
    fn normalize_slice(data: &mut [Self], scale: f64) {
        if data.is_empty() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::sync::OnceLock;
            static HAS_AVX: OnceLock<bool> = OnceLock::new();
            if *HAS_AVX.get_or_init(|| std::is_x86_feature_detected!("avx")) {
                // SAFETY: AVX confirmed at runtime.
                unsafe { return normalize_precise_avx(data, scale) };
            }
        }
        for v in data.iter_mut() {
            *v *= scale;
        }
    }
}

impl sealed::Sealed for Complex32 {}
impl NormalizeSlice for Complex32 {
    type Scale = f32;
    #[inline]
    fn normalize_slice(data: &mut [Self], scale: f32) {
        if data.is_empty() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::sync::OnceLock;
            static HAS_AVX: OnceLock<bool> = OnceLock::new();
            if *HAS_AVX.get_or_init(|| std::is_x86_feature_detected!("avx")) {
                // SAFETY: AVX confirmed at runtime.
                unsafe { return normalize_reduced_avx(data, scale) };
            }
        }
        for v in data.iter_mut() {
            *v *= scale;
        }
    }
}

// ── Entry-points ──────────────────────────────────────────────────────────────

/// Scale every element of `data` in-place by `scale`.
///
/// Dispatches to the best available implementation via `NormalizeSlice`.
/// For `Complex32` and `Complex64` this selects the AVX broadcast-multiply
/// kernel at runtime. Use `normalize_scalar` for generic element types.
#[inline]
pub(crate) fn normalize_inplace<T: NormalizeSlice>(data: &mut [T], scale: T::Scale) {
    T::normalize_slice(data, scale);
}

/// Scalar fallback for element types without a dedicated SIMD path.
///
/// Used by generic callers (e.g. `radix_composite` with `Complex<F>` where `F`
/// is a type parameter). Autovectorized by LLVM in release builds.
#[inline]
pub(crate) fn normalize_scalar<T, S: Copy>(data: &mut [T], scale: S)
where
    T: std::ops::MulAssign<S>,
{
    for v in data.iter_mut() {
        *v *= scale;
    }
}
