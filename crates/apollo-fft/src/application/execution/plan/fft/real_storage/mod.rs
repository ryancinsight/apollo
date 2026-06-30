//! Generic real-storage FFT dispatch.
//!
//! Defines the [`RealFftData`] trait that maps a real-domain storage scalar
//! (`f64`, `f32`, `f16`) to the complex spectrum produced by the underlying
//! plan. The plan scalar's native complex type `Complex<PlanScalar>` *is* the
//! spectrum type, so every transform method lives here as a single canonical
//! default body; storage scalars define only the two boundary conversions.
//! Each storage scalar's impl lives in its own submodule for SRP isolation.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::dimension_1d::{FftPlan1D, StaticFftPlan1D};
use crate::application::execution::plan::fft::dimension_2d::{FftPlan2D, StaticFftPlan2D};
use crate::application::execution::plan::fft::dimension_3d::{FftPlan3D, StaticFftPlan3D};
use leto::{Array1, Array2, Array3};
use eunomia::Complex;

mod compact;
pub(super) mod fill;
mod precise;
mod reduced;

use fill::{fill_real, fill_spectrum};

/// Real-domain storage type supported by Apollo FFT plans.
///
/// The spectrum element type is always the plan scalar's native complex type:
/// `f64` storage uses `Complex<f64>`, while lower-precision storage variants
/// (`f32`, `f16`) use `Complex<f32>`. Implementors define only the two
/// storage-boundary conversions; every transform method has one canonical
/// default body, so no per-precision algorithm forks exist.
pub trait RealFftData: Copy + Send + Sync + 'static
where
    Self::PlanScalar: MixedRadixScalar<Complex = Complex<Self::PlanScalar>>,
{
    /// The scalar type used internally for the FFT plan. Its native complex
    /// element type is the spectrum type, so plan calls need no conversion.
    type PlanScalar: MixedRadixScalar;

    /// Promote a real storage value to a spectrum element with zero imaginary
    /// part. Documented storage-boundary widening (e.g. `f16` → `f32`), not an
    /// arithmetic-path cast: plan arithmetic stays in `PlanScalar`.
    fn to_spectrum(self) -> Complex<Self::PlanScalar>;

    /// Extract the real part of a spectrum element and narrow it to storage
    /// precision at the output boundary.
    fn from_spectrum(value: Complex<Self::PlanScalar>) -> Self;

    /// Forward 1D transform dispatch.
    #[must_use]
    fn forward_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
    ) -> Array1<Complex<Self::PlanScalar>> {
        let mut output = input.mapv(Self::to_spectrum);
        plan.forward_complex_inplace(&mut output);
        output
    }
    /// Inverse 1D transform dispatch.
    #[must_use]
    fn inverse_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Complex<Self::PlanScalar>>,
    ) -> Array1<Self> {
        let mut scratch = input.clone();
        plan.inverse_complex_inplace(&mut scratch);
        scratch.mapv(Self::from_spectrum)
    }
    /// Forward 1D transform into caller-owned spectrum storage.
    fn forward_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
        output: &mut Array1<Complex<Self::PlanScalar>>,
    ) {
        fill_spectrum(input, output);
        plan.forward_complex_inplace(output);
    }
    /// Inverse 1D transform into caller-owned real storage and scratch spectrum.
    fn inverse_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Complex<Self::PlanScalar>>,
        output: &mut Array1<Self>,
        scratch: &mut Array1<Complex<Self::PlanScalar>>,
    ) {
        scratch.assign(&input.view());
        plan.inverse_complex_inplace(scratch);
        fill_real(scratch, output);
    }
    /// Inverse 1D transform into caller-owned real storage, reusing the
    /// mutable spectrum as scratch.
    fn inverse_1d_spectrum_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        spectrum: &mut Array1<Complex<Self::PlanScalar>>,
        output: &mut Array1<Self>,
    ) {
        plan.inverse_complex_inplace(spectrum);
        fill_real(spectrum, output);
    }
    /// Forward 1D transform from a real slice into owned spectrum storage.
    #[must_use]
    fn forward_1d_slice_owned(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &[Self],
    ) -> Vec<Complex<Self::PlanScalar>> {
        let mut output: Vec<_> = input.iter().map(|&value| value.to_spectrum()).collect();
        plan.forward_complex_slice_inplace(&mut output);
        output
    }
    /// Inverse 1D transform from a spectrum slice into owned real storage.
    #[must_use]
    fn inverse_1d_slice_owned(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &[Complex<Self::PlanScalar>],
    ) -> Vec<Self> {
        let mut scratch = input.to_owned();
        plan.inverse_complex_slice_inplace(&mut scratch);
        scratch.into_iter().map(Self::from_spectrum).collect()
    }
    /// Forward 1D transform into caller-owned spectrum storage using a
    /// compile-time-known length and zero-sized static plan.
    fn forward_1d_static_into<const N: usize>(
        input: &Array1<Self>,
        output: &mut Array1<Complex<Self::PlanScalar>>,
    ) {
        fill_spectrum(input, output);
        StaticFftPlan1D::<Self::PlanScalar, N>::new().forward_complex_inplace(output);
    }
    /// Inverse 1D transform into caller-owned real storage and scratch spectrum
    /// using a compile-time-known length and zero-sized static plan.
    fn inverse_1d_static_into<const N: usize>(
        input: &Array1<Complex<Self::PlanScalar>>,
        output: &mut Array1<Self>,
        scratch: &mut Array1<Complex<Self::PlanScalar>>,
    ) {
        scratch.assign(&input.view());
        StaticFftPlan1D::<Self::PlanScalar, N>::new().inverse_complex_inplace(scratch);
        fill_real(scratch, output);
    }

    /// Forward 2D transform dispatch.
    #[must_use]
    fn forward_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
    ) -> Array2<Complex<Self::PlanScalar>>
    where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        let mut out = input.mapv(Self::to_spectrum);
        plan.forward_complex_inplace(&mut out);
        out
    }
    /// Inverse 2D transform dispatch.
    #[must_use]
    fn inverse_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Complex<Self::PlanScalar>>,
    ) -> Array2<Self>
    where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        let mut out = input.clone();
        plan.inverse_complex_inplace(&mut out);
        out.mapv(Self::from_spectrum)
    }
    /// Forward 2D transform into caller-owned spectrum storage.
    fn forward_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
        output: &mut Array2<Complex<Self::PlanScalar>>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        fill_spectrum(input, output);
        plan.forward_complex_inplace(output);
    }
    /// Inverse 2D transform into caller-owned real storage and scratch spectrum.
    fn inverse_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Complex<Self::PlanScalar>>,
        output: &mut Array2<Self>,
        scratch: &mut Array2<Complex<Self::PlanScalar>>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        scratch.assign(&input.view());
        plan.inverse_complex_inplace(scratch);
        fill_real(scratch, output);
    }
    /// Inverse 2D transform into caller-owned real storage, reusing the
    /// mutable spectrum as scratch.
    fn inverse_2d_spectrum_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        spectrum: &mut Array2<Complex<Self::PlanScalar>>,
        output: &mut Array2<Self>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        plan.inverse_complex_inplace(spectrum);
        fill_real(spectrum, output);
    }
    /// Forward 2D transform into caller-owned spectrum storage using a
    /// compile-time-known shape and zero-sized static plan.
    fn forward_2d_static_into<const NX: usize, const NY: usize>(
        input: &Array2<Self>,
        output: &mut Array2<Complex<Self::PlanScalar>>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        fill_spectrum(input, output);
        StaticFftPlan2D::<Self::PlanScalar, NX, NY>::new().forward_complex_inplace(output);
    }
    /// Inverse 2D transform into caller-owned real storage and scratch spectrum
    /// using a compile-time-known shape and zero-sized static plan.
    fn inverse_2d_static_into<const NX: usize, const NY: usize>(
        input: &Array2<Complex<Self::PlanScalar>>,
        output: &mut Array2<Self>,
        scratch: &mut Array2<Complex<Self::PlanScalar>>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        scratch.assign(&input.view());
        StaticFftPlan2D::<Self::PlanScalar, NX, NY>::new().inverse_complex_inplace(scratch);
        fill_real(scratch, output);
    }

    /// Forward 3D transform dispatch.
    #[must_use]
    fn forward_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
    ) -> Array3<Complex<Self::PlanScalar>>
    where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        let mut out = input.mapv(Self::to_spectrum);
        plan.forward_complex_inplace(&mut out);
        out
    }
    /// Inverse 3D transform dispatch.
    #[must_use]
    fn inverse_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Complex<Self::PlanScalar>>,
    ) -> Array3<Self>
    where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        let mut out = input.clone();
        plan.inverse_complex_inplace(&mut out);
        out.mapv(Self::from_spectrum)
    }
    /// Forward 3D transform into caller-owned spectrum storage.
    fn forward_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
        output: &mut Array3<Complex<Self::PlanScalar>>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        fill_spectrum(input, output);
        plan.forward_complex_inplace(output);
    }
    /// Inverse 3D transform into caller-owned real storage and scratch spectrum.
    fn inverse_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Complex<Self::PlanScalar>>,
        output: &mut Array3<Self>,
        scratch: &mut Array3<Complex<Self::PlanScalar>>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        scratch.assign(&input.view());
        plan.inverse_complex_inplace(scratch);
        fill_real(scratch, output);
    }
    /// Inverse 3D transform into caller-owned real storage, reusing the
    /// mutable spectrum as scratch.
    fn inverse_3d_spectrum_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        spectrum: &mut Array3<Complex<Self::PlanScalar>>,
        output: &mut Array3<Self>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        plan.inverse_complex_inplace(spectrum);
        fill_real(spectrum, output);
    }
    /// Forward 3D transform into caller-owned spectrum storage using a
    /// compile-time-known shape and zero-sized static plan.
    fn forward_3d_static_into<const NX: usize, const NY: usize, const NZ: usize>(
        input: &Array3<Self>,
        output: &mut Array3<Complex<Self::PlanScalar>>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        fill_spectrum(input, output);
        StaticFftPlan3D::<Self::PlanScalar, NX, NY, NZ>::new().forward_complex_inplace(output);
    }
    /// Inverse 3D transform into caller-owned real storage and scratch spectrum
    /// using a compile-time-known shape and zero-sized static plan.
    fn inverse_3d_static_into<const NX: usize, const NY: usize, const NZ: usize>(
        input: &Array3<Complex<Self::PlanScalar>>,
        output: &mut Array3<Self>,
        scratch: &mut Array3<Complex<Self::PlanScalar>>,
    ) where
        Complex<Self::PlanScalar>: PlanScratch,
    {
        scratch.assign(&input.view());
        StaticFftPlan3D::<Self::PlanScalar, NX, NY, NZ>::new().inverse_complex_inplace(scratch);
        fill_real(scratch, output);
    }
}
