//! Forward complex FFT API functions.

use crate::application::execution::kernel::mixed_radix::dispatch::try_register_resident_storage;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::kernel::precision_bridge::run_via_complex32;
use crate::application::execution::kernel::FftPrecision;
use crate::application::execution::plan::fft::{
    dimension_1d::StaticFftPlan1D, dimension_2d::StaticFftPlan2D, dimension_3d::StaticFftPlan3D,
};
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use eunomia::Complex;
use half::f16;
use leto::{Array1, Array2, Array3};

/// Executes compact complex storage through the `f32` route for its length.
///
/// `Complex<f16>` is a storage representation rather than a native arithmetic
/// scalar on the supported CPU path, so the samples reach `f32` lanes either
/// way and the only question is which `f32` route runs them. That question is
/// already answered once, for `Complex32`, and answering it a second time here
/// is how this route came to take the cached plan at every length while its
/// own scalar had learned not to: the plan is the faster route for powers of
/// two and the slower one for composites
/// (`auto_dispatch::cached_plan`). Delegating leaves one decision site, so a
/// length class that is re-measured moves both types together.
///
/// The widening boundary stays here, so execution kernels remain independent
/// of orchestration, and the thread-local bridge scratch bounds temporary
/// storage to one transform.
#[inline]
fn execute_compact_storage<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex<f16>]) {
    if data.len() <= 1 {
        return;
    }
    if try_register_resident_storage::<Complex<f16>, INVERSE, NORMALIZE>(data) {
        return;
    }
    run_via_complex32(data, |buffer| {
        if INVERSE {
            if NORMALIZE {
                <eunomia::Complex32 as FftPrecision>::fft_inverse(buffer);
            } else {
                <eunomia::Complex32 as FftPrecision>::fft_inverse_unnorm(buffer);
            }
        } else {
            <eunomia::Complex32 as FftPrecision>::fft_forward(buffer);
        }
    });
}

impl FftPrecision for Complex<f16> {
    #[inline]
    fn fft_forward(data: &mut [Self]) {
        execute_compact_storage::<false, false>(data);
    }

    #[inline]
    fn fft_inverse(data: &mut [Self]) {
        execute_compact_storage::<true, true>(data);
    }

    #[inline]
    fn fft_inverse_unnorm(data: &mut [Self]) {
        execute_compact_storage::<true, false>(data);
    }
}

/// Forward complex 1D FFT in-place for a scalar profile selected at compile time.
pub fn fft_1d_complex_inplace<T>(data: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    T::get_1d_plan(
        Shape1D::new(data.size()).expect("fft_1d_complex_inplace requires non-zero length"),
    )
    .forward_complex_inplace(data);
}

/// Forward complex 1D FFT in-place for a compile-time-known length and scalar profile.
///
/// `T` selects the concrete scalar implementation at compile time, so `f32`
/// and `f64` callers monomorphize directly into their native kernels.
pub fn fft_1d_complex_static_inplace<T, const N: usize>(data: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    StaticFftPlan1D::<T, N>::new().forward_complex_inplace(data);
}

/// Forward complex 1D FFT of an owned buffer for a compile-time-known length
/// and scalar profile.
#[must_use]
pub fn fft_1d_complex_static<T, const N: usize>(mut field: Array1<Complex<T>>) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    fft_1d_complex_static_inplace::<T, N>(&mut field);
    field
}

/// Forward complex 1D FFT returning a new typed buffer.
#[must_use]
pub fn fft_1d_complex<T>(field: &Array1<Complex<T>>) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    let mut output = field.clone();
    fft_1d_complex_inplace::<T>(&mut output);
    output
}

/// Forward complex 1D FFT of an owned typed buffer.
#[must_use]
pub fn fft_1d_complex_owned<T>(mut field: Array1<Complex<T>>) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    fft_1d_complex_inplace::<T>(&mut field);
    field
}

/// Forward complex 1D FFT into a caller-provided typed buffer.
pub fn fft_1d_complex_into<T>(field: &Array1<Complex<T>>, out: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    debug_assert_eq!(
        field.size(),
        out.size(),
        "fft_1d_complex_into: length mismatch"
    );
    out.assign(&field.view());
    fft_1d_complex_inplace::<T>(out);
}

/// Forward complex 1D FFT into caller-provided storage for a compile-time-known
/// length and scalar profile.
pub fn fft_1d_complex_static_into<T, const N: usize>(
    field: &Array1<Complex<T>>,
    out: &mut Array1<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    debug_assert_eq!(
        field.size(),
        N,
        "fft_1d_complex_static_into: input length mismatch"
    );
    debug_assert_eq!(
        out.size(),
        N,
        "fft_1d_complex_static_into: output length mismatch"
    );
    out.assign(&field.view());
    fft_1d_complex_static_inplace::<T, N>(out);
}

/// Forward complex 2D FFT in-place for a scalar profile selected at compile time.
pub fn fft_2d_complex_inplace<T>(data: &mut Array2<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let [nx, ny] = data.shape();
    T::get_2d_plan(
        Shape2D::new(nx, ny).expect("fft_2d_complex_inplace requires non-zero dimensions"),
    )
    .forward_complex_inplace(data);
}

/// Forward complex 2D FFT in-place for a compile-time-known shape and scalar profile.
pub fn fft_2d_complex_static_inplace<T, const NX: usize, const NY: usize>(
    data: &mut Array2<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    StaticFftPlan2D::<T, NX, NY>::new().forward_complex_inplace(data);
}

/// Forward complex 2D FFT of an owned buffer for a compile-time-known shape
/// and scalar profile.
#[must_use]
pub fn fft_2d_complex_static<T, const NX: usize, const NY: usize>(
    mut field: Array2<Complex<T>>,
) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    fft_2d_complex_static_inplace::<T, NX, NY>(&mut field);
    field
}

/// Forward complex 2D FFT returning a new typed buffer.
#[must_use]
pub fn fft_2d_complex<T>(field: &Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let mut output = field.clone();
    fft_2d_complex_inplace::<T>(&mut output);
    output
}

/// Forward complex 2D FFT of an owned typed buffer.
#[must_use]
pub fn fft_2d_complex_owned<T>(mut field: Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    fft_2d_complex_inplace::<T>(&mut field);
    field
}

/// Forward complex 2D FFT into a caller-provided typed buffer.
pub fn fft_2d_complex_into<T>(field: &Array2<Complex<T>>, out: &mut Array2<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field.shape(),
        out.shape(),
        "fft_2d_complex_into: shape mismatch"
    );
    out.assign(&field.view());
    fft_2d_complex_inplace::<T>(out);
}

/// Forward complex 2D FFT into caller-provided storage for a compile-time-known
/// shape and scalar profile.
pub fn fft_2d_complex_static_into<T, const NX: usize, const NY: usize>(
    field: &Array2<Complex<T>>,
    out: &mut Array2<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field.shape(),
        [NX, NY],
        "fft_2d_complex_static_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY],
        "fft_2d_complex_static_into: output shape mismatch"
    );
    out.assign(&field.view());
    fft_2d_complex_static_inplace::<T, NX, NY>(out);
}

/// Forward complex 3D FFT in-place for a scalar profile selected at compile time.
pub fn fft_3d_complex_inplace<T>(data: &mut Array3<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let [nx, ny, nz] = data.shape();
    T::get_3d_plan(
        Shape3D::new(nx, ny, nz).expect("fft_3d_complex_inplace requires non-zero dimensions"),
    )
    .forward_complex_inplace(data);
}

/// Forward complex 3D FFT in-place for a compile-time-known shape and scalar profile.
pub fn fft_3d_complex_static_inplace<T, const NX: usize, const NY: usize, const NZ: usize>(
    data: &mut Array3<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    StaticFftPlan3D::<T, NX, NY, NZ>::new().forward_complex_inplace(data);
}

/// Forward complex 3D FFT of an owned buffer for a compile-time-known shape
/// and scalar profile.
#[must_use]
pub fn fft_3d_complex_static<T, const NX: usize, const NY: usize, const NZ: usize>(
    mut field: Array3<Complex<T>>,
) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    fft_3d_complex_static_inplace::<T, NX, NY, NZ>(&mut field);
    field
}

/// Forward complex 3D FFT returning a new typed buffer.
#[must_use]
pub fn fft_3d_complex<T>(field: &Array3<Complex<T>>) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let mut output = field.clone();
    fft_3d_complex_inplace::<T>(&mut output);
    output
}

/// Forward complex 3D FFT of an owned typed buffer.
#[must_use]
pub fn fft_3d_complex_owned<T>(mut field: Array3<Complex<T>>) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    fft_3d_complex_inplace::<T>(&mut field);
    field
}

/// Forward complex 3D FFT into a caller-provided typed buffer.
pub fn fft_3d_complex_into<T>(field: &Array3<Complex<T>>, out: &mut Array3<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field.shape(),
        out.shape(),
        "fft_3d_complex_into: shape mismatch"
    );
    out.assign(&field.view());
    fft_3d_complex_inplace::<T>(out);
}

/// Forward complex 3D FFT into caller-provided storage for a compile-time-known
/// shape and scalar profile.
pub fn fft_3d_complex_static_into<T, const NX: usize, const NY: usize, const NZ: usize>(
    field: &Array3<Complex<T>>,
    out: &mut Array3<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field.shape(),
        [NX, NY, NZ],
        "fft_3d_complex_static_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY, NZ],
        "fft_3d_complex_static_into: output shape mismatch"
    );
    out.assign(&field.view());
    fft_3d_complex_static_inplace::<T, NX, NY, NZ>(out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::fft_forward;
    use eunomia::Complex32;

    #[test]
    fn compact_storage_matches_the_cached_f32_plan() {
        for n in [64, 128, 256, 512] {
            let input: Vec<Complex<f16>> = (0..n)
                .map(|index| {
                    let index = index as f32;
                    Complex::new(
                        f16::from_f32((index * 0.017).sin()),
                        f16::from_f32((index * 0.023).cos()),
                    )
                })
                .collect();
            let mut expected = input
                .iter()
                .map(|value| Complex32::new(value.re.to_f32(), value.im.to_f32()))
                .collect::<Vec<_>>();
            <f32 as PlanCacheProvider>::get_1d_plan(
                Shape1D::new(n).expect("invariant: test lengths are non-zero"),
            )
            .forward_complex_slice_inplace(&mut expected);
            let expected = expected
                .into_iter()
                .map(|value| Complex::new(f16::from_f32(value.re), f16::from_f32(value.im)))
                .collect::<Vec<_>>();

            let mut actual = input;
            fft_forward(&mut actual);

            assert_eq!(actual, expected, "compact route differs at n={n}");
        }
    }
}
