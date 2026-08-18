//! Inverse complex FFT API functions.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::{
    dimension_1d::StaticFftPlan1D, dimension_2d::StaticFftPlan2D, dimension_3d::StaticFftPlan3D,
};
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use eunomia::Complex;
use leto::{Array1, Array2, Array3};

/// Inverse complex 1D FFT in-place for a scalar profile selected at compile time.
pub fn ifft_1d_complex_inplace<T>(data: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    T::get_1d_plan(
        Shape1D::new(data.size()).expect("ifft_1d_complex_inplace requires non-zero length"),
    )
    .inverse_complex_inplace(data);
}

/// Inverse complex 1D FFT in-place for a compile-time-known length and scalar profile.
///
/// `T` selects the concrete scalar implementation at compile time, so no
/// runtime plan lookup or dynamic dispatch is introduced.
pub fn ifft_1d_complex_static_inplace<T, const N: usize>(data: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    StaticFftPlan1D::<T, N>::new().inverse_complex_inplace(data);
}

/// Inverse complex 1D FFT of an owned buffer for a compile-time-known length
/// and scalar profile.
#[must_use]
pub fn ifft_1d_complex_static<T, const N: usize>(
    mut field_hat: Array1<Complex<T>>,
) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    ifft_1d_complex_static_inplace::<T, N>(&mut field_hat);
    field_hat
}

/// Inverse complex 1D FFT returning a new typed buffer.
#[must_use]
pub fn ifft_1d_complex<T>(field_hat: &Array1<Complex<T>>) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    let mut output = field_hat.clone();
    ifft_1d_complex_inplace::<T>(&mut output);
    output
}

/// Inverse complex 1D FFT of an owned typed buffer.
#[must_use]
pub fn ifft_1d_complex_owned<T>(mut field_hat: Array1<Complex<T>>) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    ifft_1d_complex_inplace::<T>(&mut field_hat);
    field_hat
}

/// Inverse complex 1D FFT into a caller-provided typed buffer.
pub fn ifft_1d_complex_into<T>(field_hat: &Array1<Complex<T>>, out: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    debug_assert_eq!(
        field_hat.size(),
        out.size(),
        "ifft_1d_complex_into: length mismatch"
    );
    out.assign(&field_hat.view());
    ifft_1d_complex_inplace::<T>(out);
}

/// Inverse complex 1D FFT into caller-provided storage for a compile-time-known
/// length and scalar profile.
pub fn ifft_1d_complex_static_into<T, const N: usize>(
    field_hat: &Array1<Complex<T>>,
    out: &mut Array1<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    debug_assert_eq!(
        field_hat.size(),
        N,
        "ifft_1d_complex_static_into: input length mismatch"
    );
    debug_assert_eq!(
        out.size(),
        N,
        "ifft_1d_complex_static_into: output length mismatch"
    );
    out.assign(&field_hat.view());
    ifft_1d_complex_static_inplace::<T, N>(out);
}

/// Inverse complex 2D FFT in-place for a scalar profile selected at compile time.
pub fn ifft_2d_complex_inplace<T>(data: &mut Array2<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let [nx, ny] = data.shape();
    T::get_2d_plan(
        Shape2D::new(nx, ny).expect("ifft_2d_complex_inplace requires non-zero dimensions"),
    )
    .inverse_complex_inplace(data);
}

/// Inverse complex 2D FFT in-place for a compile-time-known shape and scalar profile.
pub fn ifft_2d_complex_static_inplace<T, const NX: usize, const NY: usize>(
    data: &mut Array2<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    StaticFftPlan2D::<T, NX, NY>::new().inverse_complex_inplace(data);
}

/// Inverse complex 2D FFT of an owned buffer for a compile-time-known shape
/// and scalar profile.
#[must_use]
pub fn ifft_2d_complex_static<T, const NX: usize, const NY: usize>(
    mut field_hat: Array2<Complex<T>>,
) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    ifft_2d_complex_static_inplace::<T, NX, NY>(&mut field_hat);
    field_hat
}

/// Inverse complex 2D FFT returning a new typed buffer.
#[must_use]
pub fn ifft_2d_complex<T>(field_hat: &Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let mut output = field_hat.clone();
    ifft_2d_complex_inplace::<T>(&mut output);
    output
}

/// Inverse complex 2D FFT of an owned typed buffer.
#[must_use]
pub fn ifft_2d_complex_owned<T>(mut field_hat: Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    ifft_2d_complex_inplace::<T>(&mut field_hat);
    field_hat
}

/// Inverse complex 2D FFT into a caller-provided typed buffer.
pub fn ifft_2d_complex_into<T>(field_hat: &Array2<Complex<T>>, out: &mut Array2<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        out.shape(),
        "ifft_2d_complex_into: shape mismatch"
    );
    out.assign(&field_hat.view());
    ifft_2d_complex_inplace::<T>(out);
}

/// Inverse complex 2D FFT into caller-provided storage for a compile-time-known
/// shape and scalar profile.
pub fn ifft_2d_complex_static_into<T, const NX: usize, const NY: usize>(
    field_hat: &Array2<Complex<T>>,
    out: &mut Array2<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        [NX, NY],
        "ifft_2d_complex_static_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY],
        "ifft_2d_complex_static_into: output shape mismatch"
    );
    out.assign(&field_hat.view());
    ifft_2d_complex_static_inplace::<T, NX, NY>(out);
}

/// Inverse complex 3D FFT in-place for a scalar profile selected at compile time.
pub fn ifft_3d_complex_inplace<T>(data: &mut Array3<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let [nx, ny, nz] = data.shape();
    T::get_3d_plan(
        Shape3D::new(nx, ny, nz).expect("ifft_3d_complex_inplace requires non-zero dimensions"),
    )
    .inverse_complex_inplace(data);
}

/// Inverse complex 3D FFT in-place for a compile-time-known shape and scalar profile.
pub fn ifft_3d_complex_static_inplace<T, const NX: usize, const NY: usize, const NZ: usize>(
    data: &mut Array3<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    StaticFftPlan3D::<T, NX, NY, NZ>::new().inverse_complex_inplace(data);
}

/// Inverse complex 3D FFT of an owned buffer for a compile-time-known shape
/// and scalar profile.
#[must_use]
pub fn ifft_3d_complex_static<T, const NX: usize, const NY: usize, const NZ: usize>(
    mut field_hat: Array3<Complex<T>>,
) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    ifft_3d_complex_static_inplace::<T, NX, NY, NZ>(&mut field_hat);
    field_hat
}

/// Inverse complex 3D FFT returning a new typed buffer.
#[must_use]
pub fn ifft_3d_complex<T>(field_hat: &Array3<Complex<T>>) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let mut output = field_hat.clone();
    ifft_3d_complex_inplace::<T>(&mut output);
    output
}

/// Inverse complex 3D FFT of an owned typed buffer.
#[must_use]
pub fn ifft_3d_complex_owned<T>(mut field_hat: Array3<Complex<T>>) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    ifft_3d_complex_inplace::<T>(&mut field_hat);
    field_hat
}

/// Inverse complex 3D FFT into a caller-provided typed buffer.
pub fn ifft_3d_complex_into<T>(field_hat: &Array3<Complex<T>>, out: &mut Array3<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        out.shape(),
        "ifft_3d_complex_into: shape mismatch"
    );
    out.assign(&field_hat.view());
    ifft_3d_complex_inplace::<T>(out);
}

/// Inverse complex 3D FFT into caller-provided storage for a compile-time-known
/// shape and scalar profile.
pub fn ifft_3d_complex_static_into<T, const NX: usize, const NY: usize, const NZ: usize>(
    field_hat: &Array3<Complex<T>>,
    out: &mut Array3<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        [NX, NY, NZ],
        "ifft_3d_complex_static_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY, NZ],
        "ifft_3d_complex_static_into: output shape mismatch"
    );
    out.assign(&field_hat.view());
    ifft_3d_complex_static_inplace::<T, NX, NY, NZ>(out);
}
