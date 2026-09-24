//! Shared normalization primitive for radix and Rader kernel modules.
//!
//! ## Contents
//!
//! - `NormalizeSlice`: sealed trait abstracting the AVX-dispatched scale pass.
//! - `normalize_inplace`: SSOT 1/N scale pass, used by all inverse paths.

use eunomia::layout::try_cast_slice_mut;
use eunomia::{Complex, FloatElement, Pod};

/// View a complex slice as its interleaved `re, im` components.
///
/// A real scale multiplies both components alike, so the pass is one
/// contiguous run of reals; the flat view is what lets the loop vectorizer
/// emit a packed broadcast multiply instead of one scalar multiply per
/// component.
#[inline]
fn components_mut<F: FloatElement + Pod>(data: &mut [Complex<F>]) -> &mut [F] {
    try_cast_slice_mut(data).expect(
        "invariant: Complex<F> is #[repr(C)] { re, im }, so it spans two F with F's alignment",
    )
}

/// Scale `data` in place by `scale` inside an AVX frame.
///
/// One body serves every element width: the frame licenses the 256-bit
/// registers and the vectorizer picks the lane count from `F`.
/// Monomorphization never raises the enabled feature set, so the attribute
/// is required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
fn normalize_avx<F: FloatElement + Pod>(data: &mut [Complex<F>], scale: F) {
    for x in components_mut(data) {
        *x *= scale;
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Sealed normalization trait over the crate's complex elements.
///
/// Dispatches to the AVX-framed broadcast multiply on x86_64 when AVX is
/// available at runtime; falls back to the scalar loop otherwise.
pub(crate) trait NormalizeSlice: sealed::Sealed + Sized {
    type Scale: Copy;
    fn normalize_slice(data: &mut [Self], scale: Self::Scale);
}

impl<F: FloatElement + Pod> sealed::Sealed for Complex<F> {}

/// One implementation for every complex element: the real scale is the
/// element's own component type, so the multiply runs at native precision.
impl<F: FloatElement + Pod> NormalizeSlice for Complex<F> {
    type Scale = F;
    #[inline]
    fn normalize_slice(data: &mut [Self], scale: F) {
        if data.is_empty() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::sync::OnceLock;
            static HAS_AVX: OnceLock<bool> = OnceLock::new();
            if *HAS_AVX.get_or_init(|| std::is_x86_feature_detected!("avx")) {
                // SAFETY: AVX confirmed at runtime.
                unsafe { return normalize_avx(data, scale) };
            }
        }
        for x in components_mut(data) {
            *x *= scale;
        }
    }
}

/// Scale every element of `data` in-place by `scale`.
///
/// Dispatches to the best available implementation via `NormalizeSlice`,
/// which selects the AVX-framed broadcast multiply at runtime. Use
/// `normalize_scalar` when the scale is not the element's component type.
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

#[cfg(test)]
mod tests {
    use super::normalize_inplace;
    use eunomia::{CastFrom, Complex, FloatElement, Pod, F16};

    /// The dispatched pass is one multiply per component, so every lane and
    /// every tail element must equal the element's own `z * scale` bit for
    /// bit, at each length class the 256-bit stream splits differently:
    /// empty, shorter than one register, whole registers, and ragged tails.
    fn matches_per_element_scale<F>()
    where
        F: FloatElement + Pod,
        f64: CastFrom<F>,
    {
        let scale = F::from_f64(0.1);
        for len in [0_usize, 1, 2, 3, 4, 5, 8, 9, 17, 64] {
            let input: Vec<Complex<F>> = (0..len)
                .map(|i| {
                    Complex::new(
                        F::from_f64(i as f64 - 3.5),
                        F::from_f64(0.25 * i as f64 + 1.0),
                    )
                })
                .collect();
            let mut data = input.clone();
            normalize_inplace(&mut data, scale);
            for (index, (&got, &z)) in data.iter().zip(&input).enumerate() {
                let want = z * scale;
                assert_eq!(
                    (f64::cast_from(got.re), f64::cast_from(got.im)),
                    (f64::cast_from(want.re), f64::cast_from(want.im)),
                    "length {len}, element {index}"
                );
            }
        }
    }

    #[test]
    fn scaled_slice_matches_per_element_scale() {
        matches_per_element_scale::<f64>();
        matches_per_element_scale::<f32>();
        matches_per_element_scale::<F16>();
    }
}
