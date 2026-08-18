//! Forward complex FFT API functions.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::{
    dimension_1d::StaticFftPlan1D, dimension_2d::StaticFftPlan2D, dimension_3d::StaticFftPlan3D,
};
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use eunomia::Complex;
use leto::{Array1, Array2, Array3};

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
