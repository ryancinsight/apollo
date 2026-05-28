#![warn(missing_docs)]
// ── Pedantic suppressions ────────────────────────────────────────────────────
// FFT math inherently uses index-to-float casts for normalisation factors and
// twiddle-factor computation. Grid sizes are bounded by available memory
// (< 2^52), so precision loss and truncation are hypothetical, not real.
// Naming conventions in signal processing (n_x / n_y, coeff_re / coeff_im)
// are standardised in the literature; renaming them reduces clarity.
// Complex FFT plans necessarily carry many boolean precision-mode flags;
// bitset refactors would add complexity without improving safety.
// These suppressions mirror those already configured in the apollo sub-workspace
// Cargo.toml (`similar_names = "allow"`, `too_many_lines = "allow"`, etc.).
#![allow(
    clippy::cast_possible_truncation, // grid sizes < 2^24 for f32, < 2^52 for f64
    clippy::cast_precision_loss,      // usize→f32/f64 normalisation, bounded by memory
    clippy::cast_sign_loss,           // non-negative index arithmetic
    clippy::cast_possible_wrap,       // modular butterfly arithmetic
    clippy::similar_names,            // n_x/n_y/n_z, coeff_re/coeff_im — math convention
    clippy::too_many_lines,           // FFT plan builders are inherently long
    clippy::missing_panics_doc,       // cache helpers panic only on logic error / OOM
    clippy::missing_errors_doc,       // error paths documented inline in struct fields
    clippy::missing_fields_in_debug,  // manual Debug omits large internal buffers by design
    clippy::struct_excessive_bools,   // PrecisionProfile flags are orthogonal bit fields
    clippy::needless_pass_by_value,          // Copy-sized plan/shape types passed by value intentionally
    clippy::missing_const_for_thread_local,  // all thread_local! initializers already use const { }
)]
//! Apollo core crate.
//!
//! This crate owns the reusable CPU FFT implementation, shared shape and error
//! contracts, backend abstractions, and cache-backed convenience helpers.

/// Application-layer execution and orchestration.
pub mod application;
pub mod domain;
/// Infrastructure adapters.
pub mod infrastructure;

#[cfg(test)]
mod lib_tests;

pub use application::execution::plan::fft::{
    dimension_1d::FftPlan1D, dimension_2d::FftPlan2D, dimension_3d::FftPlan3D,
    real_storage::RealFftData,
};
pub use application::orchestration::cache::plans::PlanCacheProvider;
pub use domain::contracts::backend::FftBackend;
pub use domain::contracts::error::{ApolloError, ApolloResult};
pub use domain::metadata::precision::{
    BackendKind, ComputePrecision, Normalization, PrecisionMode, PrecisionProfile, StoragePrecision,
};
pub use domain::metadata::shape::{HalfSpectrum3D, Shape1D, Shape2D, Shape3D};
pub use half::f16;
pub use infrastructure::transport::cpu::CpuBackend;

pub use num_complex::Complex32;
pub use num_complex::Complex64;

pub use application::utilities::freq::{fftfreq, rfftfreq};
pub use application::utilities::shift::{fftshift, ifftshift, fftshift_inplace, ifftshift_inplace};

use ndarray::{Array1, Array2, Array3, Zip};

/// Forward 1D FFT of a real signal.
#[must_use]
pub fn fft_1d_array(field: &Array1<f64>) -> Array1<Complex64> {
    <f64 as RealFftData>::forward_1d(
        f64::get_1d_plan(Shape1D::new(field.len()).expect("fft_1d_array requires non-zero length"))
            .as_ref(),
        field,
    )
}

/// Forward 1D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_1d_array_typed<T>(field: &Array1<T>) -> Array1<T::Spectrum>
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    T::forward_1d(
        T::get_1d_plan(
            Shape1D::new(field.len()).expect("fft_1d_array_typed requires non-zero length"),
        )
        .as_ref(),
        field,
    )
}

/// Forward 2D FFT of a real array.
#[must_use]
pub fn fft_2d_array(field: &Array2<f64>) -> Array2<Complex64> {
    let (nx, ny) = field.dim();
    <f64 as RealFftData>::forward_2d(
        f64::get_2d_plan(Shape2D::new(nx, ny).expect("fft_2d_array requires non-zero dimensions"))
            .as_ref(),
        field,
    )
}

/// Forward 2D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_2d_array_typed<T>(field: &Array2<T>) -> Array2<T::Spectrum>
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny) = field.dim();
    T::forward_2d(
        T::get_2d_plan(
            Shape2D::new(nx, ny).expect("fft_2d_array_typed requires non-zero dimensions"),
        )
        .as_ref(),
        field,
    )
}

/// Forward 3D FFT of a real array.
#[must_use]
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex64> {
    let (nx, ny, nz) = field.dim();
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
pub fn fft_3d_array_typed<T>(field: &Array3<T>) -> Array3<T::Spectrum>
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny, nz) = field.dim();
    T::forward_3d(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("fft_3d_array_typed requires non-zero dimensions"),
        )
        .as_ref(),
        field,
    )
}

/// Forward 3D FFT of a real array into caller-owned typed spectrum storage.
pub fn fft_3d_array_typed_into<T>(field: &Array3<T>, out: &mut Array3<T::Spectrum>)
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny, nz) = field.dim();
    T::forward_3d_into(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("fft_3d_array_typed_into requires non-zero dimensions"),
        )
        .as_ref(),
        field,
        out,
    )
}

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

/// Forward complex 1D FFT in-place.
pub fn fft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    f64::get_1d_plan(
        Shape1D::new(data.len()).expect("fft_1d_complex_inplace requires non-zero length"),
    )
    .forward_complex_inplace(data);
}

/// Inverse complex 1D FFT in-place with FFTW-compatible normalization.
pub fn ifft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    f64::get_1d_plan(
        Shape1D::new(data.len()).expect("ifft_1d_complex_inplace requires non-zero length"),
    )
    .inverse_complex_inplace(data);
}

/// Forward complex 1D FFT returning a new buffer.
#[must_use]
pub fn fft_1d_complex(field: &Array1<Complex64>) -> Array1<Complex64> {
    let mut output = field.clone();
    fft_1d_complex_inplace(&mut output);
    output
}

/// Inverse complex 1D FFT returning a new buffer.
#[must_use]
pub fn ifft_1d_complex(field_hat: &Array1<Complex64>) -> Array1<Complex64> {
    let mut output = field_hat.clone();
    ifft_1d_complex_inplace(&mut output);
    output
}

/// Forward complex 2D FFT in-place.
pub fn fft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    let (nx, ny) = data.dim();
    f64::get_2d_plan(
        Shape2D::new(nx, ny).expect("fft_2d_complex_inplace requires non-zero dimensions"),
    )
    .forward_complex_inplace(data);
}

/// Inverse complex 2D FFT in-place with FFTW-compatible normalization.
pub fn ifft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    let (nx, ny) = data.dim();
    f64::get_2d_plan(
        Shape2D::new(nx, ny).expect("ifft_2d_complex_inplace requires non-zero dimensions"),
    )
    .inverse_complex_inplace(data);
}

/// Forward complex 2D FFT returning a new buffer.
#[must_use]
pub fn fft_2d_complex(field: &Array2<Complex64>) -> Array2<Complex64> {
    let mut output = field.clone();
    fft_2d_complex_inplace(&mut output);
    output
}

/// Inverse complex 2D FFT returning a new buffer.
#[must_use]
pub fn ifft_2d_complex(field_hat: &Array2<Complex64>) -> Array2<Complex64> {
    let mut output = field_hat.clone();
    ifft_2d_complex_inplace(&mut output);
    output
}

/// Forward complex 3D FFT in-place.
pub fn fft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    let (nx, ny, nz) = data.dim();
    f64::get_3d_plan(
        Shape3D::new(nx, ny, nz).expect("fft_3d_complex_inplace requires non-zero dimensions"),
    )
    .forward_complex_inplace(data);
}

/// Inverse complex 3D FFT in-place with FFTW-compatible normalization.
pub fn ifft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    let (nx, ny, nz) = data.dim();
    f64::get_3d_plan(
        Shape3D::new(nx, ny, nz).expect("ifft_3d_complex_inplace requires non-zero dimensions"),
    )
    .inverse_complex_inplace(data);
}

/// Forward complex 3D FFT returning a new buffer.
#[must_use]
pub fn fft_3d_complex(field: &Array3<Complex64>) -> Array3<Complex64> {
    let mut output = field.clone();
    fft_3d_complex_inplace(&mut output);
    output
}

/// Inverse complex 3D FFT returning a new buffer.
#[must_use]
pub fn ifft_3d_complex(field_hat: &Array3<Complex64>) -> Array3<Complex64> {
    let mut output = field_hat.clone();
    ifft_3d_complex_inplace(&mut output);
    output
}

/// Forward 3D FFT of a real array into a caller-provided complex buffer.
pub fn fft_3d_array_into(field: &Array3<f64>, out: &mut Array3<Complex64>) {
    debug_assert_eq!(field.dim(), out.dim(), "fft_3d_array_into: shape mismatch");
    Zip::from(out.view_mut())
        .and(field)
        .par_for_each(|dst, &src| {
            *dst = Complex64::new(src, 0.0);
        });
    let (nx, ny, nz) = field.dim();
    f64::get_3d_plan(
        Shape3D::new(nx, ny, nz).expect("fft_3d_array_into requires non-zero dimensions"),
    )
    .forward_complex_inplace(out);
}

/// Inverse 3D FFT of a complex array into a caller-provided real buffer.
pub fn ifft_3d_array_into(field_hat: &mut Array3<Complex64>, out: &mut Array3<f64>) {
    let (nx, ny, nz) = field_hat.dim();
    debug_assert_eq!(
        out.dim(),
        (nx, ny, nz),
        "ifft_3d_array_into: shape mismatch"
    );
    f64::get_3d_plan(
        Shape3D::new(nx, ny, nz).expect("ifft_3d_array_into requires non-zero dimensions"),
    )
    .inverse_complex_inplace(field_hat);
    Zip::from(out.view_mut())
        .and(field_hat.view())
        .par_for_each(|dst, src| *dst = src.re);
}

/// Forward 3D FFT of a complex array into a caller-provided complex buffer.
pub fn fft_3d_complex_into(field: &Array3<Complex64>, out: &mut Array3<Complex64>) {
    out.assign(field);
    fft_3d_complex_inplace(out);
}
