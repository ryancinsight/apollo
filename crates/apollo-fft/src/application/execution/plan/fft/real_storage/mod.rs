//! Generic real-storage FFT dispatch.
//!
//! Defines the [`RealFftData`] trait that maps a real-domain storage scalar
//! (`f64`, `f32`, `f16`) to the complex spectrum produced by the underlying
//! plan. Each storage scalar lives in its own submodule for SRP isolation;
//! shared `fill_*` helpers live in the private `fill` submodule.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::dimension_1d::FftPlan1D;
use crate::application::execution::plan::fft::dimension_2d::FftPlan2D;
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use ndarray::{Array1, Array2, Array3};

mod compact;
pub(super) mod fill;
mod precise;
mod reduced;

/// Real-domain storage type supported by Apollo FFT plans.
///
/// The associated spectrum type matches the storage family chosen by the
/// backend/profile. `f64` uses `Complex64`, while lower-precision storage
/// variants use `Complex32`.
pub trait RealFftData: Clone + Send + Sync + 'static {
    /// Complex spectrum element type produced by this storage type.
    type Spectrum: Clone + Send + Sync + 'static;

    /// The scalar type used internally for the FFT plan.
    type PlanScalar: MixedRadixScalar;

    /// Forward 1D transform dispatch.
    fn forward_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
    ) -> Array1<Self::Spectrum>;
    /// Inverse 1D transform dispatch.
    fn inverse_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self::Spectrum>,
    ) -> Array1<Self>;
    /// Forward 1D transform into caller-owned spectrum storage.
    fn forward_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
        output: &mut Array1<Self::Spectrum>,
    );
    /// Inverse 1D transform into caller-owned real storage and scratch spectrum.
    fn inverse_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self::Spectrum>,
        output: &mut Array1<Self>,
        scratch: &mut Array1<Self::Spectrum>,
    );
    /// Inverse 1D transform into caller-owned real storage, reusing the
    /// mutable spectrum as scratch.
    fn inverse_1d_spectrum_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        spectrum: &mut Array1<Self::Spectrum>,
        output: &mut Array1<Self>,
    );
    /// Forward 1D transform into caller-owned spectrum storage using a
    /// compile-time-known length and zero-sized static plan.
    fn forward_1d_static_into<const N: usize>(
        input: &Array1<Self>,
        output: &mut Array1<Self::Spectrum>,
    );
    /// Inverse 1D transform into caller-owned real storage and scratch spectrum
    /// using a compile-time-known length and zero-sized static plan.
    fn inverse_1d_static_into<const N: usize>(
        input: &Array1<Self::Spectrum>,
        output: &mut Array1<Self>,
        scratch: &mut Array1<Self::Spectrum>,
    );

    /// Forward 2D transform dispatch.
    fn forward_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
    ) -> Array2<Self::Spectrum>;
    /// Inverse 2D transform dispatch.
    fn inverse_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self::Spectrum>,
    ) -> Array2<Self>;
    /// Forward 2D transform into caller-owned spectrum storage.
    fn forward_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
        output: &mut Array2<Self::Spectrum>,
    );
    /// Inverse 2D transform into caller-owned real storage and scratch spectrum.
    fn inverse_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self::Spectrum>,
        output: &mut Array2<Self>,
        scratch: &mut Array2<Self::Spectrum>,
    );
    /// Inverse 2D transform into caller-owned real storage, reusing the
    /// mutable spectrum as scratch.
    fn inverse_2d_spectrum_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        spectrum: &mut Array2<Self::Spectrum>,
        output: &mut Array2<Self>,
    );
    /// Forward 2D transform into caller-owned spectrum storage using a
    /// compile-time-known shape and zero-sized static plan.
    fn forward_2d_static_into<const NX: usize, const NY: usize>(
        input: &Array2<Self>,
        output: &mut Array2<Self::Spectrum>,
    );
    /// Inverse 2D transform into caller-owned real storage and scratch spectrum
    /// using a compile-time-known shape and zero-sized static plan.
    fn inverse_2d_static_into<const NX: usize, const NY: usize>(
        input: &Array2<Self::Spectrum>,
        output: &mut Array2<Self>,
        scratch: &mut Array2<Self::Spectrum>,
    );

    /// Forward 3D transform dispatch.
    fn forward_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
    ) -> Array3<Self::Spectrum>;
    /// Inverse 3D transform dispatch.
    fn inverse_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self::Spectrum>,
    ) -> Array3<Self>;
    /// Forward 3D transform into caller-owned spectrum storage.
    fn forward_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
        output: &mut Array3<Self::Spectrum>,
    );
    /// Inverse 3D transform into caller-owned real storage and scratch spectrum.
    fn inverse_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self::Spectrum>,
        output: &mut Array3<Self>,
        scratch: &mut Array3<Self::Spectrum>,
    );
    /// Inverse 3D transform into caller-owned real storage, reusing the
    /// mutable spectrum as scratch.
    fn inverse_3d_spectrum_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        spectrum: &mut Array3<Self::Spectrum>,
        output: &mut Array3<Self>,
    );
    /// Forward 3D transform into caller-owned spectrum storage using a
    /// compile-time-known shape and zero-sized static plan.
    fn forward_3d_static_into<const NX: usize, const NY: usize, const NZ: usize>(
        input: &Array3<Self>,
        output: &mut Array3<Self::Spectrum>,
    );
    /// Inverse 3D transform into caller-owned real storage and scratch spectrum
    /// using a compile-time-known shape and zero-sized static plan.
    fn inverse_3d_static_into<const NX: usize, const NY: usize, const NZ: usize>(
        input: &Array3<Self::Spectrum>,
        output: &mut Array3<Self>,
        scratch: &mut Array3<Self::Spectrum>,
    );
}
