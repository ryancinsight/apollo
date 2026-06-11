//! Inverse real FFT API functions.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use crate::application::utilities::leto_interop::{
    view1_cow, array2_from_view, array3_from_view, try_array2_from_ndarray, try_array3_from_ndarray,
};

/// Inverse 1D FFT of a complex signal.
#[must_use]
pub fn ifft_1d_array(field_hat: &Array1<Complex64>) -> Array1<f64> {
    <f64 as RealFftData>::inverse_1d(
        f64::get_1d_plan(
            Shape1D::new(field_hat.len()).expect("ifft_1d_array requires non-zero length"),
        )
        .as_ref(),
        field_hat,
    )
}

/// Inverse 1D FFT of a complex spectrum using generic storage dispatch.
#[must_use]
pub fn ifft_1d_array_typed<T>(field_hat: &Array1<T::Spectrum>) -> Array1<T>
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    T::inverse_1d(
        T::get_1d_plan(
            Shape1D::new(field_hat.len()).expect("ifft_1d_array_typed requires non-zero length"),
        )
        .as_ref(),
        field_hat,
    )
}

/// Inverse 1D FFT into caller-owned real storage and scratch complex storage.
pub fn ifft_1d_array_into(
    field_hat: &Array1<Complex64>,
    out: &mut Array1<f64>,
    scratch: &mut Array1<Complex64>,
) {
    <f64 as RealFftData>::inverse_1d_into(
        f64::get_1d_plan(
            Shape1D::new(field_hat.len()).expect("ifft_1d_array_into requires non-zero length"),
        )
        .as_ref(),
        field_hat,
        out,
        scratch,
    );
}

/// Inverse 1D FFT into caller-owned real storage, reusing the mutable spectrum
/// as scratch.
///
/// This mutates `field_hat`.
pub fn ifft_1d_array_into_spectrum_scratch(
    field_hat: &mut Array1<Complex64>,
    out: &mut Array1<f64>,
) {
    ifft_1d_array_typed_into_spectrum_scratch::<f64>(field_hat, out);
}

/// Inverse 1D FFT into caller-owned typed real storage, reusing the mutable
/// typed spectrum as scratch.
///
/// This mutates `field_hat`.
pub fn ifft_1d_array_typed_into_spectrum_scratch<T>(
    field_hat: &mut Array1<T::Spectrum>,
    out: &mut Array1<T>,
) where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    debug_assert_eq!(
        field_hat.len(),
        out.len(),
        "ifft_1d_array_typed_into_spectrum_scratch: length mismatch"
    );
    T::inverse_1d_spectrum_into(
        T::get_1d_plan(
            Shape1D::new(field_hat.len())
                .expect("ifft_1d_array_typed_into_spectrum_scratch requires non-zero length"),
        )
        .as_ref(),
        field_hat,
        out,
    );
}

/// Inverse 1D FFT into caller-owned typed real storage and typed scratch spectrum.
pub fn ifft_1d_array_typed_into<T>(
    field_hat: &Array1<T::Spectrum>,
    out: &mut Array1<T>,
    scratch: &mut Array1<T::Spectrum>,
) where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    T::inverse_1d_into(
        T::get_1d_plan(
            Shape1D::new(field_hat.len())
                .expect("ifft_1d_array_typed_into requires non-zero length"),
        )
        .as_ref(),
        field_hat,
        out,
        scratch,
    );
}

/// Inverse 1D FFT into caller-owned real storage and scratch complex storage
/// for a compile-time-known length.
pub fn ifft_1d_array_static_into<const N: usize>(
    field_hat: &Array1<Complex64>,
    out: &mut Array1<f64>,
    scratch: &mut Array1<Complex64>,
) {
    ifft_1d_array_static_typed_into::<f64, N>(field_hat, out, scratch);
}

/// Inverse 1D FFT into caller-owned typed real storage and typed scratch
/// spectrum for a compile-time-known length.
pub fn ifft_1d_array_static_typed_into<T, const N: usize>(
    field_hat: &Array1<T::Spectrum>,
    out: &mut Array1<T>,
    scratch: &mut Array1<T::Spectrum>,
) where
    T: RealFftData,
{
    debug_assert_eq!(
        field_hat.len(),
        N,
        "ifft_1d_array_static_typed_into: input length mismatch"
    );
    debug_assert_eq!(
        out.len(),
        N,
        "ifft_1d_array_static_typed_into: output length mismatch"
    );
    debug_assert_eq!(
        scratch.len(),
        N,
        "ifft_1d_array_static_typed_into: scratch length mismatch"
    );
    T::inverse_1d_static_into::<N>(field_hat, out, scratch);
}

/// Inverse 1D FFT of a complex spectrum slice, returning an owned `Vec` signal.
///
/// Slice/`Vec`-based wrapper for callers that do not depend on `ndarray`.
#[must_use]
pub fn ifft_1d_slice_typed<T>(spectrum: &[T::Spectrum]) -> Vec<T>
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    T::inverse_1d_slice_owned(
        T::get_1d_plan(
            Shape1D::new(spectrum.len()).expect("ifft_1d_slice_typed requires non-zero length"),
        )
        .as_ref(),
        spectrum,
    )
}

/// Inverse 1D FFT of a Leto spectrum view, returning Mnemosyne-backed Leto storage.
#[must_use]
pub fn ifft_1d_leto(
    field_hat: leto::ArrayView1<'_, Complex64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 1> {
    ifft_1d_leto_typed::<f64>(field_hat)
}

/// Inverse 1D FFT of a Leto spectrum view using generic storage dispatch.
///
/// C-contiguous Leto views are consumed through a borrowed slice. Strided views
/// are copied once in logical row-major order before entering the existing IFFT
/// slice boundary. The returned Leto array is backed by Mnemosyne allocation.
#[must_use]
pub fn ifft_1d_leto_typed<T>(
    field_hat: leto::ArrayView1<'_, T::Spectrum>,
) -> leto::Array<T, leto::MnemosyneStorage<T>, 1>
where
    T: RealFftData + PlanCacheProvider + Copy,
    T::Spectrum: Copy,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let spectrum = view1_cow(&field_hat);
    let signal = ifft_1d_slice_typed::<T>(&spectrum);
    leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice([signal.len()], &signal)
        .expect("IFFT signal length must match Leto output shape")
}

/// Inverse 2D FFT of a complex array.
#[must_use]
pub fn ifft_2d_array(field_hat: &Array2<Complex64>) -> Array2<f64> {
    let (nx, ny) = field_hat.dim();
    <f64 as RealFftData>::inverse_2d(
        f64::get_2d_plan(Shape2D::new(nx, ny).expect("ifft_2d_array requires non-zero dimensions"))
            .as_ref(),
        field_hat,
    )
}

/// Inverse 2D FFT of a complex spectrum using generic storage dispatch.
#[must_use]
pub fn ifft_2d_array_typed<T>(field_hat: &Array2<T::Spectrum>) -> Array2<T>
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny) = field_hat.dim();
    T::inverse_2d(
        T::get_2d_plan(
            Shape2D::new(nx, ny).expect("ifft_2d_array_typed requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
    )
}

/// Inverse 2D FFT into caller-owned real storage and scratch complex storage.
pub fn ifft_2d_array_into(
    field_hat: &Array2<Complex64>,
    out: &mut Array2<f64>,
    scratch: &mut Array2<Complex64>,
) {
    let (nx, ny) = field_hat.dim();
    <f64 as RealFftData>::inverse_2d_into(
        f64::get_2d_plan(
            Shape2D::new(nx, ny).expect("ifft_2d_array_into requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
        scratch,
    );
}

/// Inverse 2D FFT into caller-owned real storage, reusing the mutable spectrum
/// as scratch.
///
/// This mutates `field_hat`.
pub fn ifft_2d_array_into_spectrum_scratch(
    field_hat: &mut Array2<Complex64>,
    out: &mut Array2<f64>,
) {
    ifft_2d_array_typed_into_spectrum_scratch::<f64>(field_hat, out);
}

/// Inverse 2D FFT into caller-owned typed real storage, reusing the mutable
/// typed spectrum as scratch.
///
/// This mutates `field_hat`.
pub fn ifft_2d_array_typed_into_spectrum_scratch<T>(
    field_hat: &mut Array2<T::Spectrum>,
    out: &mut Array2<T>,
) where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny) = field_hat.dim();
    debug_assert_eq!(
        out.dim(),
        (nx, ny),
        "ifft_2d_array_typed_into_spectrum_scratch: shape mismatch"
    );
    T::inverse_2d_spectrum_into(
        T::get_2d_plan(
            Shape2D::new(nx, ny)
                .expect("ifft_2d_array_typed_into_spectrum_scratch requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
    );
}

/// Inverse 2D FFT into caller-owned typed real storage and typed scratch spectrum.
pub fn ifft_2d_array_typed_into<T>(
    field_hat: &Array2<T::Spectrum>,
    out: &mut Array2<T>,
    scratch: &mut Array2<T::Spectrum>,
) where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny) = field_hat.dim();
    T::inverse_2d_into(
        T::get_2d_plan(
            Shape2D::new(nx, ny).expect("ifft_2d_array_typed_into requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
        scratch,
    );
}

/// Inverse 2D FFT into caller-owned real storage and scratch complex storage
/// for a compile-time-known shape.
pub fn ifft_2d_array_static_into<const NX: usize, const NY: usize>(
    field_hat: &Array2<Complex64>,
    out: &mut Array2<f64>,
    scratch: &mut Array2<Complex64>,
) {
    ifft_2d_array_static_typed_into::<f64, NX, NY>(field_hat, out, scratch);
}

/// Inverse 2D FFT into caller-owned typed real storage and typed scratch
/// spectrum for a compile-time-known shape.
pub fn ifft_2d_array_static_typed_into<T, const NX: usize, const NY: usize>(
    field_hat: &Array2<T::Spectrum>,
    out: &mut Array2<T>,
    scratch: &mut Array2<T::Spectrum>,
) where
    T: RealFftData,
{
    debug_assert_eq!(
        field_hat.dim(),
        (NX, NY),
        "ifft_2d_array_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.dim(),
        (NX, NY),
        "ifft_2d_array_static_typed_into: output shape mismatch"
    );
    debug_assert_eq!(
        scratch.dim(),
        (NX, NY),
        "ifft_2d_array_static_typed_into: scratch shape mismatch"
    );
    T::inverse_2d_static_into::<NX, NY>(field_hat, out, scratch);
}

/// Inverse 3D FFT of a complex array.
#[must_use]
pub fn ifft_3d_array(field_hat: &Array3<Complex64>) -> Array3<f64> {
    let (nx, ny, nz) = field_hat.dim();
    <f64 as RealFftData>::inverse_3d(
        f64::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("ifft_3d_array requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
    )
}

/// Inverse 3D FFT of a complex spectrum using generic storage dispatch.
#[must_use]
pub fn ifft_3d_array_typed<T>(field_hat: &Array3<T::Spectrum>) -> Array3<T>
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny, nz) = field_hat.dim();
    T::inverse_3d(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("ifft_3d_array_typed requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
    )
}

/// Inverse 3D FFT into caller-owned typed real storage and typed scratch spectrum.
pub fn ifft_3d_array_typed_into<T>(
    field_hat: &Array3<T::Spectrum>,
    out: &mut Array3<T>,
    scratch: &mut Array3<T::Spectrum>,
) where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny, nz) = field_hat.dim();
    T::inverse_3d_into(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz)
                .expect("ifft_3d_array_typed_into requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
        scratch,
    )
}

/// Inverse 3D FFT into caller-owned typed real storage, reusing the mutable
/// typed spectrum as scratch.
///
/// This mutates `field_hat`.
pub fn ifft_3d_array_typed_into_spectrum_scratch<T>(
    field_hat: &mut Array3<T::Spectrum>,
    out: &mut Array3<T>,
) where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny, nz) = field_hat.dim();
    debug_assert_eq!(
        out.dim(),
        (nx, ny, nz),
        "ifft_3d_array_typed_into_spectrum_scratch: shape mismatch"
    );
    T::inverse_3d_spectrum_into(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz)
                .expect("ifft_3d_array_typed_into_spectrum_scratch requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
    );
}

/// Inverse 3D FFT into caller-owned real storage and scratch complex storage
/// for a compile-time-known shape.
pub fn ifft_3d_array_static_into<const NX: usize, const NY: usize, const NZ: usize>(
    field_hat: &Array3<Complex64>,
    out: &mut Array3<f64>,
    scratch: &mut Array3<Complex64>,
) {
    ifft_3d_array_static_typed_into::<f64, NX, NY, NZ>(field_hat, out, scratch);
}

/// Inverse 3D FFT into caller-owned typed real storage and typed scratch
/// spectrum for a compile-time-known shape.
pub fn ifft_3d_array_static_typed_into<T, const NX: usize, const NY: usize, const NZ: usize>(
    field_hat: &Array3<T::Spectrum>,
    out: &mut Array3<T>,
    scratch: &mut Array3<T::Spectrum>,
) where
    T: RealFftData,
{
    debug_assert_eq!(
        field_hat.dim(),
        (NX, NY, NZ),
        "ifft_3d_array_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.dim(),
        (NX, NY, NZ),
        "ifft_3d_array_static_typed_into: output shape mismatch"
    );
    debug_assert_eq!(
        scratch.dim(),
        (NX, NY, NZ),
        "ifft_3d_array_static_typed_into: scratch shape mismatch"
    );
    T::inverse_3d_static_into::<NX, NY, NZ>(field_hat, out, scratch);
}

/// Inverse 3D FFT of a complex array into a caller-provided real buffer.
pub fn ifft_3d_array_into(field_hat: &mut Array3<Complex64>, out: &mut Array3<f64>) {
    ifft_3d_array_typed_into_spectrum_scratch::<f64>(field_hat, out);
}

/// Inverse 3D FFT into caller-owned real storage and scratch complex storage.
pub fn ifft_3d_array_into_scratch(
    field_hat: &Array3<Complex64>,
    out: &mut Array3<f64>,
    scratch: &mut Array3<Complex64>,
) {
    ifft_3d_array_typed_into::<f64>(field_hat, out, scratch);
}

/// Inverse 2D FFT of a Leto spectrum view, returning Mnemosyne-backed Leto storage.
#[must_use]
pub fn ifft_2d_leto(
    field_hat: leto::ArrayView2<'_, Complex64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 2> {
    ifft_2d_leto_typed::<f64>(field_hat)
}

/// Inverse 2D FFT of a Leto spectrum view using generic storage dispatch.
#[must_use]
pub fn ifft_2d_leto_typed<T>(
    field_hat: leto::ArrayView2<'_, T::Spectrum>,
) -> leto::Array<T, leto::MnemosyneStorage<T>, 2>
where
    T: RealFftData + PlanCacheProvider + Copy,
    T::Spectrum: Copy,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let nd_array = array2_from_view(&field_hat);
    let output = ifft_2d_array_typed::<T>(&nd_array);
    try_array2_from_ndarray(&output)
        .expect("IFFT signal shape must match Leto output shape")
}

/// Inverse 3D FFT of a Leto spectrum view, returning Mnemosyne-backed Leto storage.
#[must_use]
pub fn ifft_3d_leto(
    field_hat: leto::ArrayView3<'_, Complex64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 3> {
    ifft_3d_leto_typed::<f64>(field_hat)
}

/// Inverse 3D FFT of a Leto spectrum view using generic storage dispatch.
#[must_use]
pub fn ifft_3d_leto_typed<T>(
    field_hat: leto::ArrayView3<'_, T::Spectrum>,
) -> leto::Array<T, leto::MnemosyneStorage<T>, 3>
where
    T: RealFftData + PlanCacheProvider + Copy,
    T::Spectrum: Copy,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let nd_array = array3_from_view(&field_hat);
    let output = ifft_3d_array_typed::<T>(&nd_array);
    try_array3_from_ndarray(&output)
        .expect("IFFT signal shape must match Leto output shape")
}
