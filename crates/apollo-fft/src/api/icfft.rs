//! Inverse complex FFT API functions.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::{
    dimension_1d::StaticFftPlan1D, dimension_2d::StaticFftPlan2D, dimension_3d::StaticFftPlan3D,
};
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use leto::{Array1, Array2, Array3};
use eunomia::{Complex, Complex64};

/// Inverse complex 1D FFT in-place with FFTW-compatible normalization.
pub fn ifft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    ifft_1d_complex_typed_inplace::<f64>(data);
}

/// Inverse complex 1D FFT in-place for a scalar profile selected at compile time.
pub fn ifft_1d_complex_typed_inplace<T>(data: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    T::get_1d_plan(
        Shape1D::new(data.size()).expect("ifft_1d_complex_typed_inplace requires non-zero length"),
    )
    .inverse_complex_inplace(data);
}

/// Inverse complex 1D FFT in-place for a compile-time-known length.
///
/// The length is encoded in `N`, so execution uses the zero-sized static plan
/// path rather than runtime plan lookup.
pub fn ifft_1d_complex_static_inplace<const N: usize>(data: &mut Array1<Complex64>) {
    ifft_1d_complex_static_typed_inplace::<f64, N>(data);
}

/// Inverse complex 1D FFT in-place for a compile-time-known length and scalar profile.
///
/// `T` selects the concrete scalar implementation at compile time, so no
/// runtime plan lookup or dynamic dispatch is introduced.
pub fn ifft_1d_complex_static_typed_inplace<T, const N: usize>(data: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    StaticFftPlan1D::<T, N>::new().inverse_complex_inplace(data);
}

/// Inverse complex 1D FFT of an owned buffer for a compile-time-known length.
///
/// This consumes the input, transforms it in place through the zero-sized
/// static plan, and returns the same allocation.
#[must_use]
pub fn ifft_1d_complex_static<const N: usize>(field_hat: Array1<Complex64>) -> Array1<Complex64> {
    ifft_1d_complex_static_typed::<f64, N>(field_hat)
}

/// Inverse complex 1D FFT of an owned buffer for a compile-time-known length
/// and scalar profile.
#[must_use]
pub fn ifft_1d_complex_static_typed<T, const N: usize>(
    mut field_hat: Array1<Complex<T>>,
) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    ifft_1d_complex_static_typed_inplace::<T, N>(&mut field_hat);
    field_hat
}

/// Inverse complex 1D FFT returning a new buffer.
#[must_use]
pub fn ifft_1d_complex(field_hat: &Array1<Complex64>) -> Array1<Complex64> {
    ifft_1d_complex_typed(field_hat)
}

/// Inverse complex 1D FFT returning a new typed buffer.
#[must_use]
pub fn ifft_1d_complex_typed<T>(field_hat: &Array1<Complex<T>>) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    let mut output = field_hat.clone();
    ifft_1d_complex_typed_inplace::<T>(&mut output);
    output
}

/// Inverse complex 1D FFT of an owned buffer.
///
/// This consumes the input, transforms it in place, and returns the same
/// allocation.
#[must_use]
pub fn ifft_1d_complex_owned(field_hat: Array1<Complex64>) -> Array1<Complex64> {
    ifft_1d_complex_typed_owned::<f64>(field_hat)
}

/// Inverse complex 1D FFT of an owned typed buffer.
#[must_use]
pub fn ifft_1d_complex_typed_owned<T>(mut field_hat: Array1<Complex<T>>) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    ifft_1d_complex_typed_inplace::<T>(&mut field_hat);
    field_hat
}

/// Inverse complex 1D FFT into a caller-provided complex buffer.
pub fn ifft_1d_complex_into(field_hat: &Array1<Complex64>, out: &mut Array1<Complex64>) {
    ifft_1d_complex_typed_into::<f64>(field_hat, out);
}

/// Inverse complex 1D FFT into a caller-provided typed buffer.
pub fn ifft_1d_complex_typed_into<T>(field_hat: &Array1<Complex<T>>, out: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    debug_assert_eq!(
        field_hat.size(),
        out.size(),
        "ifft_1d_complex_typed_into: length mismatch"
    );
    out.assign(field_hat);
    ifft_1d_complex_typed_inplace::<T>(out);
}

/// Inverse complex 1D FFT into caller-provided storage for a compile-time-known length.
///
/// This copies the input into `out`, then executes through the zero-sized
/// static plan path without runtime plan lookup.
pub fn ifft_1d_complex_static_into<const N: usize>(
    field_hat: &Array1<Complex64>,
    out: &mut Array1<Complex64>,
) {
    ifft_1d_complex_static_typed_into::<f64, N>(field_hat, out);
}

/// Inverse complex 1D FFT into caller-provided storage for a compile-time-known
/// length and scalar profile.
pub fn ifft_1d_complex_static_typed_into<T, const N: usize>(
    field_hat: &Array1<Complex<T>>,
    out: &mut Array1<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    debug_assert_eq!(
        field_hat.size(),
        N,
        "ifft_1d_complex_static_typed_into: input length mismatch"
    );
    debug_assert_eq!(
        out.size(),
        N,
        "ifft_1d_complex_static_typed_into: output length mismatch"
    );
    out.assign(field_hat);
    ifft_1d_complex_static_typed_inplace::<T, N>(out);
}

/// Inverse complex 2D FFT in-place with FFTW-compatible normalization.
pub fn ifft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    ifft_2d_complex_typed_inplace::<f64>(data);
}

/// Inverse complex 2D FFT in-place for a scalar profile selected at compile time.
pub fn ifft_2d_complex_typed_inplace<T>(data: &mut Array2<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let [nx, ny] = data.shape();
    T::get_2d_plan(
        Shape2D::new(nx, ny).expect("ifft_2d_complex_typed_inplace requires non-zero dimensions"),
    )
    .inverse_complex_inplace(data);
}

/// Inverse complex 2D FFT in-place for a compile-time-known shape.
///
/// Both axes are encoded as const generics and route through zero-sized static
/// plans rather than runtime plan lookup.
pub fn ifft_2d_complex_static_inplace<const NX: usize, const NY: usize>(
    data: &mut Array2<Complex64>,
) {
    ifft_2d_complex_static_typed_inplace::<f64, NX, NY>(data);
}

/// Inverse complex 2D FFT in-place for a compile-time-known shape and scalar profile.
pub fn ifft_2d_complex_static_typed_inplace<T, const NX: usize, const NY: usize>(
    data: &mut Array2<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    StaticFftPlan2D::<T, NX, NY>::new().inverse_complex_inplace(data);
}

/// Inverse complex 2D FFT of an owned buffer for a compile-time-known shape.
///
/// This consumes the input, transforms it in place through the zero-sized
/// static plan, and returns the same allocation.
#[must_use]
pub fn ifft_2d_complex_static<const NX: usize, const NY: usize>(
    field_hat: Array2<Complex64>,
) -> Array2<Complex64> {
    ifft_2d_complex_static_typed::<f64, NX, NY>(field_hat)
}

/// Inverse complex 2D FFT of an owned buffer for a compile-time-known shape
/// and scalar profile.
#[must_use]
pub fn ifft_2d_complex_static_typed<T, const NX: usize, const NY: usize>(
    mut field_hat: Array2<Complex<T>>,
) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    ifft_2d_complex_static_typed_inplace::<T, NX, NY>(&mut field_hat);
    field_hat
}

/// Inverse complex 2D FFT returning a new buffer.
#[must_use]
pub fn ifft_2d_complex(field_hat: &Array2<Complex64>) -> Array2<Complex64> {
    ifft_2d_complex_typed(field_hat)
}

/// Inverse complex 2D FFT returning a new typed buffer.
#[must_use]
pub fn ifft_2d_complex_typed<T>(field_hat: &Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let mut output = field_hat.clone();
    ifft_2d_complex_typed_inplace::<T>(&mut output);
    output
}

/// Inverse complex 2D FFT of an owned buffer.
///
/// This consumes the input, transforms it in place, and returns the same
/// allocation.
#[must_use]
pub fn ifft_2d_complex_owned(field_hat: Array2<Complex64>) -> Array2<Complex64> {
    ifft_2d_complex_typed_owned::<f64>(field_hat)
}

/// Inverse complex 2D FFT of an owned typed buffer.
#[must_use]
pub fn ifft_2d_complex_typed_owned<T>(mut field_hat: Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    ifft_2d_complex_typed_inplace::<T>(&mut field_hat);
    field_hat
}

/// Inverse complex 2D FFT into a caller-provided complex buffer.
pub fn ifft_2d_complex_into(field_hat: &Array2<Complex64>, out: &mut Array2<Complex64>) {
    ifft_2d_complex_typed_into::<f64>(field_hat, out);
}

/// Inverse complex 2D FFT into a caller-provided typed buffer.
pub fn ifft_2d_complex_typed_into<T>(field_hat: &Array2<Complex<T>>, out: &mut Array2<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        out.shape(),
        "ifft_2d_complex_typed_into: shape mismatch"
    );
    out.assign(field_hat);
    ifft_2d_complex_typed_inplace::<T>(out);
}

/// Inverse complex 2D FFT into caller-provided storage for a compile-time-known shape.
///
/// This copies the input into `out`, then executes through the zero-sized
/// static plan path without runtime plan lookup.
pub fn ifft_2d_complex_static_into<const NX: usize, const NY: usize>(
    field_hat: &Array2<Complex64>,
    out: &mut Array2<Complex64>,
) {
    ifft_2d_complex_static_typed_into::<f64, NX, NY>(field_hat, out);
}

/// Inverse complex 2D FFT into caller-provided storage for a compile-time-known
/// shape and scalar profile.
pub fn ifft_2d_complex_static_typed_into<T, const NX: usize, const NY: usize>(
    field_hat: &Array2<Complex<T>>,
    out: &mut Array2<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        (NX, NY),
        "ifft_2d_complex_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        (NX, NY),
        "ifft_2d_complex_static_typed_into: output shape mismatch"
    );
    out.assign(field_hat);
    ifft_2d_complex_static_typed_inplace::<T, NX, NY>(out);
}

/// Inverse complex 3D FFT in-place with FFTW-compatible normalization.
pub fn ifft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    ifft_3d_complex_typed_inplace::<f64>(data);
}

/// Inverse complex 3D FFT in-place for a scalar profile selected at compile time.
pub fn ifft_3d_complex_typed_inplace<T>(data: &mut Array3<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let [nx, ny, nz] = data.shape();
    T::get_3d_plan(
        Shape3D::new(nx, ny, nz)
            .expect("ifft_3d_complex_typed_inplace requires non-zero dimensions"),
    )
    .inverse_complex_inplace(data);
}

/// Inverse complex 3D FFT in-place for a compile-time-known shape.
///
/// All axes are encoded as const generics and route through zero-sized static
/// plans rather than runtime plan lookup.
pub fn ifft_3d_complex_static_inplace<const NX: usize, const NY: usize, const NZ: usize>(
    data: &mut Array3<Complex64>,
) {
    ifft_3d_complex_static_typed_inplace::<f64, NX, NY, NZ>(data);
}

/// Inverse complex 3D FFT in-place for a compile-time-known shape and scalar profile.
pub fn ifft_3d_complex_static_typed_inplace<T, const NX: usize, const NY: usize, const NZ: usize>(
    data: &mut Array3<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    StaticFftPlan3D::<T, NX, NY, NZ>::new().inverse_complex_inplace(data);
}

/// Inverse complex 3D FFT of an owned buffer for a compile-time-known shape.
///
/// This consumes the input, transforms it in place through the zero-sized
/// static plan, and returns the same allocation.
#[must_use]
pub fn ifft_3d_complex_static<const NX: usize, const NY: usize, const NZ: usize>(
    field_hat: Array3<Complex64>,
) -> Array3<Complex64> {
    ifft_3d_complex_static_typed::<f64, NX, NY, NZ>(field_hat)
}

/// Inverse complex 3D FFT of an owned buffer for a compile-time-known shape
/// and scalar profile.
#[must_use]
pub fn ifft_3d_complex_static_typed<T, const NX: usize, const NY: usize, const NZ: usize>(
    mut field_hat: Array3<Complex<T>>,
) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    ifft_3d_complex_static_typed_inplace::<T, NX, NY, NZ>(&mut field_hat);
    field_hat
}

/// Inverse complex 3D FFT returning a new buffer.
#[must_use]
pub fn ifft_3d_complex(field_hat: &Array3<Complex64>) -> Array3<Complex64> {
    ifft_3d_complex_typed(field_hat)
}

/// Inverse complex 3D FFT returning a new typed buffer.
#[must_use]
pub fn ifft_3d_complex_typed<T>(field_hat: &Array3<Complex<T>>) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let mut output = field_hat.clone();
    ifft_3d_complex_typed_inplace::<T>(&mut output);
    output
}

/// Inverse complex 3D FFT of an owned buffer.
///
/// This consumes the input, transforms it in place, and returns the same
/// allocation.
#[must_use]
pub fn ifft_3d_complex_owned(field_hat: Array3<Complex64>) -> Array3<Complex64> {
    ifft_3d_complex_typed_owned::<f64>(field_hat)
}

/// Inverse complex 3D FFT of an owned typed buffer.
#[must_use]
pub fn ifft_3d_complex_typed_owned<T>(mut field_hat: Array3<Complex<T>>) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    ifft_3d_complex_typed_inplace::<T>(&mut field_hat);
    field_hat
}

/// Inverse complex 3D FFT into a caller-provided complex buffer.
pub fn ifft_3d_complex_into(field_hat: &Array3<Complex64>, out: &mut Array3<Complex64>) {
    ifft_3d_complex_typed_into::<f64>(field_hat, out);
}

/// Inverse complex 3D FFT into a caller-provided typed buffer.
pub fn ifft_3d_complex_typed_into<T>(field_hat: &Array3<Complex<T>>, out: &mut Array3<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        out.shape(),
        "ifft_3d_complex_typed_into: shape mismatch"
    );
    out.assign(field_hat);
    ifft_3d_complex_typed_inplace::<T>(out);
}

/// Inverse complex 3D FFT into caller-provided storage for a compile-time-known shape.
///
/// This copies the input into `out`, then executes through the zero-sized
/// static plan path without runtime plan lookup.
pub fn ifft_3d_complex_static_into<const NX: usize, const NY: usize, const NZ: usize>(
    field_hat: &Array3<Complex64>,
    out: &mut Array3<Complex64>,
) {
    ifft_3d_complex_static_typed_into::<f64, NX, NY, NZ>(field_hat, out);
}

/// Inverse complex 3D FFT into caller-provided storage for a compile-time-known
/// shape and scalar profile.
pub fn ifft_3d_complex_static_typed_into<T, const NX: usize, const NY: usize, const NZ: usize>(
    field_hat: &Array3<Complex<T>>,
    out: &mut Array3<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        (NX, NY, NZ),
        "ifft_3d_complex_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        (NX, NY, NZ),
        "ifft_3d_complex_static_typed_into: output shape mismatch"
    );
    out.assign(field_hat);
    ifft_3d_complex_static_typed_inplace::<T, NX, NY, NZ>(out);
}
