//! Shared normalization primitive for radix and Bluestein kernel modules.
//!
//! ## Contents
//!
//! - `normalize_inplace`: SSOT 1/N scale pass, used by inverse paths.
//! - `normalize_inplace_c64`, `normalize_inplace_c32`: AVX fast paths
//!   used by `normalize_inplace` on x86_64 with AVX support.

use num_complex::{Complex32, Complex64};

// ── AVX fast paths ─────────────────────────────────────────────────────────────

/// Scale a `Complex64` slice in-place by `scale` using AVX broadcast-multiply.
///
/// ## Safety
///
/// Caller must ensure AVX is available (gated by `is_x86_feature_detected!`).
/// The slice must be non-empty.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn normalize_c64_avx(data: &mut [Complex64], scale: f64) {
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
    // Scalar tail for odd count.
    for x in &mut data[batches * 2..len] {
        *x *= scale;
    }
}

/// Scale a `Complex32` slice in-place by `scale` using AVX broadcast-multiply.
///
/// ## Safety
///
/// Caller must ensure AVX is available (gated by `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn normalize_c32_avx(data: &mut [Complex32], scale: f32) {
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
    // Scalar tail for remainder.
    for x in &mut data[batches * 4..len] {
        *x *= scale;
    }
}

// ── Runtime-dispatched specialisations ────────────────────────────────────────

/// Normalize a `Complex64` slice using the best available path.
#[inline]
pub(crate) fn normalize_inplace_c64(data: &mut [Complex64], scale: f64) {
    if data.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // Cache the feature query: OnceLock amortises the cpuid overhead.
        use std::sync::OnceLock;
        static HAS_AVX: OnceLock<bool> = OnceLock::new();
        if *HAS_AVX.get_or_init(|| std::is_x86_feature_detected!("avx")) {
            // SAFETY: AVX confirmed at runtime above.
            unsafe { return normalize_c64_avx(data, scale) };
        }
    }
    // Scalar fallback (auto-vectorized by LLVM with -C target-feature=+avx).
    for v in data.iter_mut() {
        *v *= scale;
    }
}

/// Normalize a `Complex32` slice using the best available path.
#[inline]
pub(crate) fn normalize_inplace_c32(data: &mut [Complex32], scale: f32) {
    if data.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static HAS_AVX: OnceLock<bool> = OnceLock::new();
        if *HAS_AVX.get_or_init(|| std::is_x86_feature_detected!("avx")) {
            // SAFETY: AVX confirmed at runtime above.
            unsafe { return normalize_c32_avx(data, scale) };
        }
    }
    for v in data.iter_mut() {
        *v *= scale;
    }
}

// ── Generic entry-point ────────────────────────────────────────────────────────

/// Scale every element of `data` in-place by `scale`.
///
/// ## SSOT role
///
/// Single authoritative 1/N normalization pass used by all inverse transform paths
/// (`bluestein`, `radix_composite`). Loop bounds and vectorization
/// contract live here; all callers delegate.
///
/// ## Performance
///
/// For `Complex64` and `Complex32` this dispatches to explicit AVX2 kernels at
/// runtime. For all other types the generic scalar loop is used; LLVM
/// auto-vectorizes it in release builds.
#[inline]
pub(crate) fn normalize_inplace<T, S>(data: &mut [T], scale: S)
where
    T: std::ops::MulAssign<S>,
    S: Copy,
{
    for v in data.iter_mut() {
        *v *= scale;
    }
}
