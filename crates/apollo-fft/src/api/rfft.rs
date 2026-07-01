//! Forward real FFT API functions.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::application::utilities::leto_interop::{
    try_dense_from_contiguous, view1_cow,
};
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use leto::{Array1, Array2, Array3};
use eunomia::Complex;
use eunomia::Complex64;

/// Forward 1D FFT of a real signal.
#[must_use]
pub fn fft_1d_array(field: &Array1<f64>) -> Array1<Complex64> {
    <f64 as RealFftData>::forward_1d(
        f64::get_1d_plan(Shape1D::new(field.size()).expect("fft_1d_array requires non-zero length"))
            .as_ref(),
        field,
    )
}

/// Forward 1D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_1d_array_typed<T>(field: &Array1<T>) -> Array1<Complex<T::PlanScalar>>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    T::forward_1d(
        T::get_1d_plan(
            Shape1D::new(field.size()).expect("fft_1d_array_typed requires non-zero length"),
        )
        .as_ref(),
        field,
    )
}

/// Forward 1D FFT of a real signal into caller-owned complex storage.
pub fn fft_1d_array_into(field: &Array1<f64>, out: &mut Array1<Complex64>) {
    <f64 as RealFftData>::forward_1d_into(
        f64::get_1d_plan(
            Shape1D::new(field.size()).expect("fft_1d_array_into requires non-zero length"),
        )
        .as_ref(),
        field,
        out,
    );
}

/// Forward 1D FFT of a real array into caller-owned typed spectrum storage.
pub fn fft_1d_array_typed_into<T>(field: &Array1<T>, out: &mut Array1<Complex<T::PlanScalar>>)
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    T::forward_1d_into(
        T::get_1d_plan(
            Shape1D::new(field.size()).expect("fft_1d_array_typed_into requires non-zero length"),
        )
        .as_ref(),
        field,
        out,
    );
}

/// Forward 1D FFT of a real signal into caller-owned complex storage for a
/// compile-time-known length.
pub fn fft_1d_array_static_into<const N: usize>(field: &Array1<f64>, out: &mut Array1<Complex64>) {
    fft_1d_array_static_typed_into::<f64, N>(field, out);
}

/// Forward 1D FFT of a real array into caller-owned typed spectrum storage for
/// a compile-time-known length.
pub fn fft_1d_array_static_typed_into<T, const N: usize>(
    field: &Array1<T>,
    out: &mut Array1<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    debug_assert_eq!(
        field.size(),
        N,
        "fft_1d_array_static_typed_into: input length mismatch"
    );
    debug_assert_eq!(
        out.size(),
        N,
        "fft_1d_array_static_typed_into: output length mismatch"
    );
    T::forward_1d_static_into::<N>(field, out);
}

/// Forward 1D FFT of a real signal slice, returning an owned `Vec` spectrum.
///
/// Slice/`Vec`-based wrapper for callers that prefer raw slices over Leto `Array` types.
#[must_use]
pub fn fft_1d_slice_typed<T>(signal: &[T]) -> Vec<Complex<T::PlanScalar>>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    T::forward_1d_slice_owned(
        T::get_1d_plan(
            Shape1D::new(signal.len()).expect("fft_1d_slice_typed requires non-zero length"),
        )
        .as_ref(),
        signal,
    )
}

/// Forward 1D FFT of a Leto real view, returning Mnemosyne-backed Leto storage.
#[must_use]
pub fn fft_1d_leto(
    field: leto::ArrayView1<'_, f64>,
) -> leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1> {
    fft_1d_leto_typed::<f64>(field)
}

/// Forward 1D FFT of a Leto real view using generic storage dispatch.
///
/// C-contiguous Leto views are consumed through a borrowed slice. Strided views
/// are copied once in logical row-major order before entering the existing FFT
/// slice boundary. The returned Leto array is backed by Mnemosyne allocation.
#[must_use]
pub fn fft_1d_leto_typed<T>(
    field: leto::ArrayView1<'_, T>,
) -> leto::Array<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 1>
where
    T: RealFftData + PlanCacheProvider + Copy,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let signal = view1_cow(&field);
    let spectrum = fft_1d_slice_typed::<T>(&signal);
    leto::Array::<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 1>::from_mnemosyne_vec(
        [spectrum.len()],
        spectrum,
    )
    .expect("FFT spectrum length must match Leto output shape")
}

/// Forward 2D FFT of a real array.
#[must_use]
pub fn fft_2d_array(field: &Array2<f64>) -> Array2<Complex64> {
    let [nx, ny] = field.shape();
    <f64 as RealFftData>::forward_2d(
        f64::get_2d_plan(Shape2D::new(nx, ny).expect("fft_2d_array requires non-zero dimensions"))
            .as_ref(),
        field,
    )
}

/// Forward 2D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_2d_array_typed<T>(field: &Array2<T>) -> Array2<Complex<T::PlanScalar>>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny] = field.shape();
    T::forward_2d(
        T::get_2d_plan(
            Shape2D::new(nx, ny).expect("fft_2d_array_typed requires non-zero dimensions"),
        )
        .as_ref(),
        field,
    )
}

/// Forward 2D FFT of a real array into caller-owned complex storage.
pub fn fft_2d_array_into(field: &Array2<f64>, out: &mut Array2<Complex64>) {
    let [nx, ny] = field.shape();
    <f64 as RealFftData>::forward_2d_into(
        f64::get_2d_plan(
            Shape2D::new(nx, ny).expect("fft_2d_array_into requires non-zero dimensions"),
        )
        .as_ref(),
        field,
        out,
    );
}

/// Forward 2D FFT of a real array into caller-owned typed spectrum storage.
pub fn fft_2d_array_typed_into<T>(field: &Array2<T>, out: &mut Array2<Complex<T::PlanScalar>>)
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny] = field.shape();
    T::forward_2d_into(
        T::get_2d_plan(
            Shape2D::new(nx, ny).expect("fft_2d_array_typed_into requires non-zero dimensions"),
        )
        .as_ref(),
        field,
        out,
    );
}

/// Forward 2D FFT of a real array into caller-owned complex storage for a
/// compile-time-known shape.
pub fn fft_2d_array_static_into<const NX: usize, const NY: usize>(
    field: &Array2<f64>,
    out: &mut Array2<Complex64>,
) {
    fft_2d_array_static_typed_into::<f64, NX, NY>(field, out);
}

/// Forward 2D FFT of a real array into caller-owned typed spectrum storage for
/// a compile-time-known shape.
pub fn fft_2d_array_static_typed_into<T, const NX: usize, const NY: usize>(
    field: &Array2<T>,
    out: &mut Array2<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    debug_assert_eq!(
        field.shape(),
        [NX, NY],
        "fft_2d_array_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY],
        "fft_2d_array_static_typed_into: output shape mismatch"
    );
    T::forward_2d_static_into::<NX, NY>(field, out);
}

/// Forward 3D FFT of a real array.
#[must_use]
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex64> {
    let [nx, ny, nz] = field.shape();
    <f64 as RealFftData>::forward_3d(
        f64::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("fft_3d_array requires non-zero dimensions"),
        )
        .as_ref(),
        field,
    )
}

/// Forward 3D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_3d_array_typed<T>(field: &Array3<T>) -> Array3<Complex<T::PlanScalar>>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny, nz] = field.shape();
    T::forward_3d(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("fft_3d_array_typed requires non-zero dimensions"),
        )
        .as_ref(),
        field,
    )
}

/// Forward 3D FFT of a real array into caller-owned typed spectrum storage.
pub fn fft_3d_array_typed_into<T>(field: &Array3<T>, out: &mut Array3<Complex<T::PlanScalar>>)
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny, nz] = field.shape();
    T::forward_3d_into(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("fft_3d_array_typed_into requires non-zero dimensions"),
        )
        .as_ref(),
        field,
        out,
    );
}

/// Forward 3D FFT of a real array into caller-owned complex storage.
pub fn fft_3d_array_into(field: &Array3<f64>, out: &mut Array3<Complex64>) {
    fft_3d_array_typed_into::<f64>(field, out);
}

/// Forward 3D FFT of a real array into caller-owned complex storage for a
/// compile-time-known shape.
pub fn fft_3d_array_static_into<const NX: usize, const NY: usize, const NZ: usize>(
    field: &Array3<f64>,
    out: &mut Array3<Complex64>,
) {
    fft_3d_array_static_typed_into::<f64, NX, NY, NZ>(field, out);
}

/// Forward 3D FFT of a real array into caller-owned typed spectrum storage for
/// a compile-time-known shape.
pub fn fft_3d_array_static_typed_into<T, const NX: usize, const NY: usize, const NZ: usize>(
    field: &Array3<T>,
    out: &mut Array3<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    debug_assert_eq!(
        field.shape(),
        [NX, NY, NZ],
        "fft_3d_array_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY, NZ],
        "fft_3d_array_static_typed_into: output shape mismatch"
    );
    T::forward_3d_static_into::<NX, NY, NZ>(field, out);
}

/// Forward 2D FFT of a Leto real view, returning Mnemosyne-backed Leto storage.
#[must_use]
pub fn fft_2d_leto(
    field: leto::ArrayView2<'_, f64>,
) -> leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2> {
    fft_2d_leto_typed::<f64>(field)
}

/// Forward 2D FFT of a Leto real view using generic storage dispatch.
#[must_use]
pub fn fft_2d_leto_typed<T>(
    field: leto::ArrayView2<'_, T>,
) -> leto::Array<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 2>
where
    T: RealFftData + PlanCacheProvider + Copy,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let dense_array = field.to_contiguous();
    let output = fft_2d_array_typed::<T>(&dense_array);
    try_dense_from_contiguous(&output).expect("FFT spectrum shape must match Leto output shape")
}

/// Forward 3D FFT of a Leto real view, returning Mnemosyne-backed Leto storage.
#[must_use]
pub fn fft_3d_leto(
    field: leto::ArrayView3<'_, f64>,
) -> leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 3> {
    fft_3d_leto_typed::<f64>(field)
}

/// Forward 3D FFT of a Leto real view using generic storage dispatch.
#[must_use]
pub fn fft_3d_leto_typed<T>(
    field: leto::ArrayView3<'_, T>,
) -> leto::Array<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 3>
where
    T: RealFftData + PlanCacheProvider + Copy,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let dense_array = field.to_contiguous();
    let output = fft_3d_array_typed::<T>(&dense_array);
    try_dense_from_contiguous(&output).expect("FFT spectrum shape must match Leto output shape")
}
