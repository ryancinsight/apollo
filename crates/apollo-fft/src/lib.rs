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
    clippy::many_single_char_names,    // FFT/Rader formulas use standard n, m, j, k, w notation
    clippy::too_many_lines,           // FFT plan builders are inherently long
    clippy::missing_panics_doc,       // cache helpers panic only on logic error / OOM
    clippy::missing_errors_doc,       // error paths documented inline in struct fields
    clippy::missing_fields_in_debug,  // manual Debug omits large internal buffers by design
    clippy::struct_excessive_bools,   // PrecisionProfile flags are orthogonal bit fields
    clippy::cast_ptr_alignment,       // loadu/storeu SIMD intrinsics intentionally accept unaligned lanes
    clippy::option_option,             // tri-state caches encode unknown/unsupported/supported distinctly
    clippy::approx_constant,           // generated tables preserve audited literal bit patterns
    clippy::needless_pass_by_value,          // Copy-sized plan/shape types passed by value intentionally
    clippy::missing_const_for_thread_local,  // all thread_local! initializers already use const { }
    clippy::excessive_precision,             // Winograd/codelet coefficients carry one guard digit past
                                             // f64 precision so the compiler selects the intended
                                             // nearest-representable value; trimming would alter
                                             // bit-exact differential-test results (e.g. -13/12 literal)
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
    dimension_1d::{FftPlan1D, StaticFftPlan1D},
    dimension_2d::{FftPlan2D, StaticFftPlan2D},
    dimension_3d::{FftPlan3D, StaticFftPlan3D},
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
pub use application::utilities::shift::{fftshift, fftshift_inplace, ifftshift, ifftshift_inplace};

use application::execution::kernel::mixed_radix::MixedRadixScalar;
use application::execution::plan::fft::workspace::PlanScratch;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;

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

/// Forward 1D FFT of a real signal into caller-owned complex storage.
pub fn fft_1d_array_into(field: &Array1<f64>, out: &mut Array1<Complex64>) {
    <f64 as RealFftData>::forward_1d_into(
        f64::get_1d_plan(
            Shape1D::new(field.len()).expect("fft_1d_array_into requires non-zero length"),
        )
        .as_ref(),
        field,
        out,
    );
}

/// Forward 1D FFT of a real array into caller-owned typed spectrum storage.
pub fn fft_1d_array_typed_into<T>(field: &Array1<T>, out: &mut Array1<T::Spectrum>)
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    T::forward_1d_into(
        T::get_1d_plan(
            Shape1D::new(field.len()).expect("fft_1d_array_typed_into requires non-zero length"),
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
    out: &mut Array1<T::Spectrum>,
) where
    T: RealFftData,
{
    debug_assert_eq!(
        field.len(),
        N,
        "fft_1d_array_static_typed_into: input length mismatch"
    );
    debug_assert_eq!(
        out.len(),
        N,
        "fft_1d_array_static_typed_into: output length mismatch"
    );
    T::forward_1d_static_into::<N>(field, out);
}

/// Forward 1D FFT of a real signal slice, returning an owned `Vec` spectrum.
///
/// Slice/`Vec`-based wrapper over [`fft_1d_array_typed`] for callers that do not
/// depend on `ndarray`; the `Array1` is constructed and consumed internally.
#[must_use]
pub fn fft_1d_slice_typed<T>(signal: &[T]) -> Vec<T::Spectrum>
where
    T: RealFftData + PlanCacheProvider + Clone,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
    T::Spectrum: Clone,
{
    fft_1d_array_typed::<T>(&Array1::from_vec(signal.to_vec())).to_vec()

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

/// Forward 2D FFT of a real array into caller-owned complex storage.
pub fn fft_2d_array_into(field: &Array2<f64>, out: &mut Array2<Complex64>) {
    let (nx, ny) = field.dim();
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
pub fn fft_2d_array_typed_into<T>(field: &Array2<T>, out: &mut Array2<T::Spectrum>)
where
    T: RealFftData + PlanCacheProvider,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let (nx, ny) = field.dim();
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
    out: &mut Array2<T::Spectrum>,
) where
    T: RealFftData,
{
    debug_assert_eq!(
        field.dim(),
        (NX, NY),
        "fft_2d_array_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.dim(),
        (NX, NY),
        "fft_2d_array_static_typed_into: output shape mismatch"
    );
    T::forward_2d_static_into::<NX, NY>(field, out);
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
    out: &mut Array3<T::Spectrum>,
) where
    T: RealFftData,
{
    debug_assert_eq!(
        field.dim(),
        (NX, NY, NZ),
        "fft_3d_array_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.dim(),
        (NX, NY, NZ),
        "fft_3d_array_static_typed_into: output shape mismatch"
    );
    T::forward_3d_static_into::<NX, NY, NZ>(field, out);
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
/// Slice/`Vec`-based wrapper over [`ifft_1d_array_typed`] for callers that do not
/// depend on `ndarray`; the `Array1` is constructed and consumed internally.
#[must_use]
pub fn ifft_1d_slice_typed<T>(spectrum: &[T::Spectrum]) -> Vec<T>
where
    T: RealFftData + PlanCacheProvider + Clone,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
    T::Spectrum: Clone,
{
    ifft_1d_array_typed::<T>(&Array1::from_vec(spectrum.to_vec())).to_vec()
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

/// Forward complex 1D FFT in-place.
pub fn fft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    fft_1d_complex_typed_inplace::<f64>(data);
}

/// Forward complex 1D FFT in-place for a scalar profile selected at compile time.
pub fn fft_1d_complex_typed_inplace<T>(data: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    T::get_1d_plan(
        Shape1D::new(data.len()).expect("fft_1d_complex_typed_inplace requires non-zero length"),
    )
    .forward_complex_inplace(data);
}

/// Forward complex 1D FFT in-place for a compile-time-known length.
///
/// The length is encoded in `N`, so execution uses the zero-sized static plan
/// path rather than runtime plan lookup.
pub fn fft_1d_complex_static_inplace<const N: usize>(data: &mut Array1<Complex64>) {
    fft_1d_complex_static_typed_inplace::<f64, N>(data);
}

/// Forward complex 1D FFT in-place for a compile-time-known length and scalar profile.
///
/// `T` selects the concrete scalar implementation at compile time, so `f32`
/// and `f64` callers monomorphize directly into their native kernels.
pub fn fft_1d_complex_static_typed_inplace<T, const N: usize>(data: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    StaticFftPlan1D::<T, N>::new().forward_complex_inplace(data);
}

/// Forward complex 1D FFT of an owned buffer for a compile-time-known length.
///
/// This consumes the input, transforms it in place through the zero-sized
/// static plan, and returns the same allocation.
#[must_use]
pub fn fft_1d_complex_static<const N: usize>(field: Array1<Complex64>) -> Array1<Complex64> {
    fft_1d_complex_static_typed::<f64, N>(field)
}

/// Forward complex 1D FFT of an owned buffer for a compile-time-known length
/// and scalar profile.
#[must_use]
pub fn fft_1d_complex_static_typed<T, const N: usize>(
    mut field: Array1<Complex<T>>,
) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    fft_1d_complex_static_typed_inplace::<T, N>(&mut field);
    field
}

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
        Shape1D::new(data.len()).expect("ifft_1d_complex_typed_inplace requires non-zero length"),
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

/// Forward complex 1D FFT returning a new buffer.
#[must_use]
pub fn fft_1d_complex(field: &Array1<Complex64>) -> Array1<Complex64> {
    fft_1d_complex_typed(field)
}

/// Forward complex 1D FFT returning a new typed buffer.
#[must_use]
pub fn fft_1d_complex_typed<T>(field: &Array1<Complex<T>>) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    let mut output = field.clone();
    fft_1d_complex_typed_inplace::<T>(&mut output);
    output
}

/// Forward complex 1D FFT of an owned buffer.
///
/// This consumes the input, transforms it in place, and returns the same
/// allocation.
#[must_use]
pub fn fft_1d_complex_owned(field: Array1<Complex64>) -> Array1<Complex64> {
    fft_1d_complex_typed_owned::<f64>(field)
}

/// Forward complex 1D FFT of an owned typed buffer.
#[must_use]
pub fn fft_1d_complex_typed_owned<T>(mut field: Array1<Complex<T>>) -> Array1<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    fft_1d_complex_typed_inplace::<T>(&mut field);
    field
}

/// Forward complex 1D FFT into a caller-provided complex buffer.
pub fn fft_1d_complex_into(field: &Array1<Complex64>, out: &mut Array1<Complex64>) {
    fft_1d_complex_typed_into::<f64>(field, out);
}

/// Forward complex 1D FFT into a caller-provided typed buffer.
pub fn fft_1d_complex_typed_into<T>(field: &Array1<Complex<T>>, out: &mut Array1<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
{
    debug_assert_eq!(
        field.len(),
        out.len(),
        "fft_1d_complex_typed_into: length mismatch"
    );
    out.assign(field);
    fft_1d_complex_typed_inplace::<T>(out);
}

/// Forward complex 1D FFT into caller-provided storage for a compile-time-known length.
///
/// This copies the input into `out`, then executes through the zero-sized
/// static plan path without runtime plan lookup.
pub fn fft_1d_complex_static_into<const N: usize>(
    field: &Array1<Complex64>,
    out: &mut Array1<Complex64>,
) {
    fft_1d_complex_static_typed_into::<f64, N>(field, out);
}

/// Forward complex 1D FFT into caller-provided storage for a compile-time-known
/// length and scalar profile.
pub fn fft_1d_complex_static_typed_into<T, const N: usize>(
    field: &Array1<Complex<T>>,
    out: &mut Array1<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    debug_assert_eq!(
        field.len(),
        N,
        "fft_1d_complex_static_typed_into: input length mismatch"
    );
    debug_assert_eq!(
        out.len(),
        N,
        "fft_1d_complex_static_typed_into: output length mismatch"
    );
    out.assign(field);
    fft_1d_complex_static_typed_inplace::<T, N>(out);
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
        field_hat.len(),
        out.len(),
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
        field_hat.len(),
        N,
        "ifft_1d_complex_static_typed_into: input length mismatch"
    );
    debug_assert_eq!(
        out.len(),
        N,
        "ifft_1d_complex_static_typed_into: output length mismatch"
    );
    out.assign(field_hat);
    ifft_1d_complex_static_typed_inplace::<T, N>(out);
}

/// Forward complex 2D FFT in-place.
pub fn fft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    fft_2d_complex_typed_inplace::<f64>(data);
}

/// Forward complex 2D FFT in-place for a scalar profile selected at compile time.
pub fn fft_2d_complex_typed_inplace<T>(data: &mut Array2<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let (nx, ny) = data.dim();
    T::get_2d_plan(
        Shape2D::new(nx, ny).expect("fft_2d_complex_typed_inplace requires non-zero dimensions"),
    )
    .forward_complex_inplace(data);
}

/// Forward complex 2D FFT in-place for a compile-time-known shape.
///
/// Both axes are encoded as const generics and route through zero-sized static
/// plans rather than runtime plan lookup.
pub fn fft_2d_complex_static_inplace<const NX: usize, const NY: usize>(
    data: &mut Array2<Complex64>,
) {
    fft_2d_complex_static_typed_inplace::<f64, NX, NY>(data);
}

/// Forward complex 2D FFT in-place for a compile-time-known shape and scalar profile.
pub fn fft_2d_complex_static_typed_inplace<T, const NX: usize, const NY: usize>(
    data: &mut Array2<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    StaticFftPlan2D::<T, NX, NY>::new().forward_complex_inplace(data);
}

/// Forward complex 2D FFT of an owned buffer for a compile-time-known shape.
///
/// This consumes the input, transforms it in place through the zero-sized
/// static plan, and returns the same allocation.
#[must_use]
pub fn fft_2d_complex_static<const NX: usize, const NY: usize>(
    field: Array2<Complex64>,
) -> Array2<Complex64> {
    fft_2d_complex_static_typed::<f64, NX, NY>(field)
}

/// Forward complex 2D FFT of an owned buffer for a compile-time-known shape
/// and scalar profile.
#[must_use]
pub fn fft_2d_complex_static_typed<T, const NX: usize, const NY: usize>(
    mut field: Array2<Complex<T>>,
) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    fft_2d_complex_static_typed_inplace::<T, NX, NY>(&mut field);
    field
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
    let (nx, ny) = data.dim();
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

/// Forward complex 2D FFT returning a new buffer.
#[must_use]
pub fn fft_2d_complex(field: &Array2<Complex64>) -> Array2<Complex64> {
    fft_2d_complex_typed(field)
}

/// Forward complex 2D FFT returning a new typed buffer.
#[must_use]
pub fn fft_2d_complex_typed<T>(field: &Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let mut output = field.clone();
    fft_2d_complex_typed_inplace::<T>(&mut output);
    output
}

/// Forward complex 2D FFT of an owned buffer.
///
/// This consumes the input, transforms it in place, and returns the same
/// allocation.
#[must_use]
pub fn fft_2d_complex_owned(field: Array2<Complex64>) -> Array2<Complex64> {
    fft_2d_complex_typed_owned::<f64>(field)
}

/// Forward complex 2D FFT of an owned typed buffer.
#[must_use]
pub fn fft_2d_complex_typed_owned<T>(mut field: Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    fft_2d_complex_typed_inplace::<T>(&mut field);
    field
}

/// Forward complex 2D FFT into a caller-provided complex buffer.
pub fn fft_2d_complex_into(field: &Array2<Complex64>, out: &mut Array2<Complex64>) {
    fft_2d_complex_typed_into::<f64>(field, out);
}

/// Forward complex 2D FFT into a caller-provided typed buffer.
pub fn fft_2d_complex_typed_into<T>(field: &Array2<Complex<T>>, out: &mut Array2<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field.dim(),
        out.dim(),
        "fft_2d_complex_typed_into: shape mismatch"
    );
    out.assign(field);
    fft_2d_complex_typed_inplace::<T>(out);
}

/// Forward complex 2D FFT into caller-provided storage for a compile-time-known shape.
///
/// This copies the input into `out`, then executes through the zero-sized
/// static plan path without runtime plan lookup.
pub fn fft_2d_complex_static_into<const NX: usize, const NY: usize>(
    field: &Array2<Complex64>,
    out: &mut Array2<Complex64>,
) {
    fft_2d_complex_static_typed_into::<f64, NX, NY>(field, out);
}

/// Forward complex 2D FFT into caller-provided storage for a compile-time-known
/// shape and scalar profile.
pub fn fft_2d_complex_static_typed_into<T, const NX: usize, const NY: usize>(
    field: &Array2<Complex<T>>,
    out: &mut Array2<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field.dim(),
        (NX, NY),
        "fft_2d_complex_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.dim(),
        (NX, NY),
        "fft_2d_complex_static_typed_into: output shape mismatch"
    );
    out.assign(field);
    fft_2d_complex_static_typed_inplace::<T, NX, NY>(out);
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
        field_hat.dim(),
        out.dim(),
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
        field_hat.dim(),
        (NX, NY),
        "ifft_2d_complex_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.dim(),
        (NX, NY),
        "ifft_2d_complex_static_typed_into: output shape mismatch"
    );
    out.assign(field_hat);
    ifft_2d_complex_static_typed_inplace::<T, NX, NY>(out);
}

/// Forward complex 3D FFT in-place.
pub fn fft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    fft_3d_complex_typed_inplace::<f64>(data);
}

/// Forward complex 3D FFT in-place for a scalar profile selected at compile time.
pub fn fft_3d_complex_typed_inplace<T>(data: &mut Array3<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let (nx, ny, nz) = data.dim();
    T::get_3d_plan(
        Shape3D::new(nx, ny, nz)
            .expect("fft_3d_complex_typed_inplace requires non-zero dimensions"),
    )
    .forward_complex_inplace(data);
}

/// Forward complex 3D FFT in-place for a compile-time-known shape.
///
/// All axes are encoded as const generics and route through zero-sized static
/// plans rather than runtime plan lookup.
pub fn fft_3d_complex_static_inplace<const NX: usize, const NY: usize, const NZ: usize>(
    data: &mut Array3<Complex64>,
) {
    fft_3d_complex_static_typed_inplace::<f64, NX, NY, NZ>(data);
}

/// Forward complex 3D FFT in-place for a compile-time-known shape and scalar profile.
pub fn fft_3d_complex_static_typed_inplace<T, const NX: usize, const NY: usize, const NZ: usize>(
    data: &mut Array3<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    StaticFftPlan3D::<T, NX, NY, NZ>::new().forward_complex_inplace(data);
}

/// Forward complex 3D FFT of an owned buffer for a compile-time-known shape.
///
/// This consumes the input, transforms it in place through the zero-sized
/// static plan, and returns the same allocation.
#[must_use]
pub fn fft_3d_complex_static<const NX: usize, const NY: usize, const NZ: usize>(
    field: Array3<Complex64>,
) -> Array3<Complex64> {
    fft_3d_complex_static_typed::<f64, NX, NY, NZ>(field)
}

/// Forward complex 3D FFT of an owned buffer for a compile-time-known shape
/// and scalar profile.
#[must_use]
pub fn fft_3d_complex_static_typed<T, const NX: usize, const NY: usize, const NZ: usize>(
    mut field: Array3<Complex<T>>,
) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    fft_3d_complex_static_typed_inplace::<T, NX, NY, NZ>(&mut field);
    field
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
    let (nx, ny, nz) = data.dim();
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

/// Forward complex 3D FFT returning a new buffer.
#[must_use]
pub fn fft_3d_complex(field: &Array3<Complex64>) -> Array3<Complex64> {
    fft_3d_complex_typed(field)
}

/// Forward complex 3D FFT returning a new typed buffer.
#[must_use]
pub fn fft_3d_complex_typed<T>(field: &Array3<Complex<T>>) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    let mut output = field.clone();
    fft_3d_complex_typed_inplace::<T>(&mut output);
    output
}

/// Forward complex 3D FFT of an owned buffer.
///
/// This consumes the input, transforms it in place, and returns the same
/// allocation.
#[must_use]
pub fn fft_3d_complex_owned(field: Array3<Complex64>) -> Array3<Complex64> {
    fft_3d_complex_typed_owned::<f64>(field)
}

/// Forward complex 3D FFT of an owned typed buffer.
#[must_use]
pub fn fft_3d_complex_typed_owned<T>(mut field: Array3<Complex<T>>) -> Array3<Complex<T>>
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    fft_3d_complex_typed_inplace::<T>(&mut field);
    field
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

/// Forward 3D FFT of a real array into a caller-provided complex buffer.
pub fn fft_3d_array_into(field: &Array3<f64>, out: &mut Array3<Complex64>) {
    fft_3d_array_typed_into::<f64>(field, out);
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

/// Forward 3D FFT of a complex array into a caller-provided complex buffer.
pub fn fft_3d_complex_into(field: &Array3<Complex64>, out: &mut Array3<Complex64>) {
    fft_3d_complex_typed_into::<f64>(field, out);
}

/// Forward complex 3D FFT into a caller-provided typed buffer.
pub fn fft_3d_complex_typed_into<T>(field: &Array3<Complex<T>>, out: &mut Array3<Complex<T>>)
where
    T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field.dim(),
        out.dim(),
        "fft_3d_complex_typed_into: shape mismatch"
    );
    out.assign(field);
    fft_3d_complex_typed_inplace::<T>(out);
}

/// Forward complex 3D FFT into caller-provided storage for a compile-time-known shape.
///
/// This copies the input into `out`, then executes through the zero-sized
/// static plan path without runtime plan lookup.
pub fn fft_3d_complex_static_into<const NX: usize, const NY: usize, const NZ: usize>(
    field: &Array3<Complex64>,
    out: &mut Array3<Complex64>,
) {
    fft_3d_complex_static_typed_into::<f64, NX, NY, NZ>(field, out);
}

/// Forward complex 3D FFT into caller-provided storage for a compile-time-known
/// shape and scalar profile.
pub fn fft_3d_complex_static_typed_into<T, const NX: usize, const NY: usize, const NZ: usize>(
    field: &Array3<Complex<T>>,
    out: &mut Array3<Complex<T>>,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
    T::Complex: PlanScratch,
{
    debug_assert_eq!(
        field.dim(),
        (NX, NY, NZ),
        "fft_3d_complex_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.dim(),
        (NX, NY, NZ),
        "fft_3d_complex_static_typed_into: output shape mismatch"
    );
    out.assign(field);
    fft_3d_complex_static_typed_inplace::<T, NX, NY, NZ>(out);
}

/// Inverse 3D FFT of a complex array into a caller-provided complex buffer.
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
        field_hat.dim(),
        out.dim(),
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
        field_hat.dim(),
        (NX, NY, NZ),
        "ifft_3d_complex_static_typed_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.dim(),
        (NX, NY, NZ),
        "ifft_3d_complex_static_typed_into: output shape mismatch"
    );
    out.assign(field_hat);
    ifft_3d_complex_static_typed_inplace::<T, NX, NY, NZ>(out);
}
