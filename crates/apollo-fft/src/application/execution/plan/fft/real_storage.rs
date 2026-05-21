//! Generic real-storage FFT dispatch.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::dimension_1d::FftPlan1D;
use crate::application::execution::plan::fft::dimension_2d::FftPlan2D;
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use half::f16;
use ndarray::{Array, Array1, Array2, Array3, Dimension, Zip};
use num_complex::{Complex32, Complex64};

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
}

#[inline]
fn fill_complex64<D>(input: &Array<f64, D>, output: &mut Array<Complex64, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "real-to-complex shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = Complex64::new(src, 0.0));
}

#[inline]
fn fill_real64<D>(input: &Array<Complex64, D>, output: &mut Array<f64, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "complex-to-real shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = src.re);
}

#[inline]
fn fill_complex32<D>(input: &Array<f32, D>, output: &mut Array<Complex32, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "real-to-complex shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = Complex32::new(src, 0.0));
}

#[inline]
fn fill_real32<D>(input: &Array<Complex32, D>, output: &mut Array<f32, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "complex-to-real shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = src.re);
}

#[inline]
fn fill_complex32_from_f16<D>(input: &Array<f16, D>, output: &mut Array<Complex32, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "real-to-complex shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = Complex32::new(src.to_f32(), 0.0));
}

#[inline]
fn fill_f16_from_complex32<D>(input: &Array<Complex32, D>, output: &mut Array<f16, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "complex-to-real shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = f16::from_f32(src.re));
}

impl RealFftData for f64 {
    type Spectrum = Complex64;
    type PlanScalar = f64;

    fn forward_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
    ) -> Array1<Self::Spectrum> {
        let mut output = input.mapv(|v| Complex64::new(v, 0.0));
        plan.forward_complex_inplace(&mut output);
        output
    }

    fn inverse_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self::Spectrum>,
    ) -> Array1<Self> {
        let mut scratch = input.clone();
        plan.inverse_complex_inplace(&mut scratch);
        scratch.mapv(|c| c.re)
    }

    fn forward_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
        output: &mut Array1<Self::Spectrum>,
    ) {
        fill_complex64(input, output);
        plan.forward_complex_inplace(output);
    }

    fn inverse_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self::Spectrum>,
        output: &mut Array1<Self>,
        scratch: &mut Array1<Self::Spectrum>,
    ) {
        scratch.assign(input);
        plan.inverse_complex_inplace(scratch);
        fill_real64(scratch, output);
    }

    fn forward_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
    ) -> Array2<Self::Spectrum> {
        let mut out = input.mapv(|v| Complex64::new(v, 0.0));
        plan.forward_complex_inplace(&mut out);
        out
    }

    fn inverse_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self::Spectrum>,
    ) -> Array2<Self> {
        let mut out = input.clone();
        plan.inverse_complex_inplace(&mut out);
        out.mapv(|c| c.re)
    }

    fn forward_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
        output: &mut Array2<Self::Spectrum>,
    ) {
        fill_complex64(input, output);
        plan.forward_complex_inplace(output);
    }

    fn inverse_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self::Spectrum>,
        output: &mut Array2<Self>,
        scratch: &mut Array2<Self::Spectrum>,
    ) {
        scratch.assign(input);
        plan.inverse_complex_inplace(scratch);
        fill_real64(scratch, output);
    }

    fn forward_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
    ) -> Array3<Self::Spectrum> {
        let mut out = input.mapv(|v| Complex64::new(v, 0.0));
        plan.forward_complex_inplace(&mut out);
        out
    }

    fn inverse_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self::Spectrum>,
    ) -> Array3<Self> {
        let mut out = input.clone();
        plan.inverse_complex_inplace(&mut out);
        out.mapv(|c| c.re)
    }

    fn forward_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
        output: &mut Array3<Self::Spectrum>,
    ) {
        fill_complex64(input, output);
        plan.forward_complex_inplace(output);
    }

    fn inverse_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self::Spectrum>,
        output: &mut Array3<Self>,
        scratch: &mut Array3<Self::Spectrum>,
    ) {
        scratch.assign(input);
        plan.inverse_complex_inplace(scratch);
        fill_real64(scratch, output);
    }
}

impl RealFftData for f32 {
    type Spectrum = Complex32;
    type PlanScalar = f32;

    fn forward_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
    ) -> Array1<Self::Spectrum> {
        let mut output = input.mapv(|v| Complex32::new(v, 0.0));
        plan.forward_complex_inplace(&mut output);
        output
    }

    fn inverse_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self::Spectrum>,
    ) -> Array1<Self> {
        let mut scratch = input.clone();
        plan.inverse_complex_inplace(&mut scratch);
        scratch.mapv(|c| c.re)
    }

    fn forward_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
        output: &mut Array1<Self::Spectrum>,
    ) {
        fill_complex32(input, output);
        plan.forward_complex_inplace(output);
    }

    fn inverse_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self::Spectrum>,
        output: &mut Array1<Self>,
        scratch: &mut Array1<Self::Spectrum>,
    ) {
        scratch.assign(input);
        plan.inverse_complex_inplace(scratch);
        fill_real32(scratch, output);
    }

    fn forward_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
    ) -> Array2<Self::Spectrum> {
        let mut out = input.mapv(|v| Complex32::new(v, 0.0));
        plan.forward_complex_inplace(&mut out);
        out
    }

    fn inverse_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self::Spectrum>,
    ) -> Array2<Self> {
        let mut out = input.clone();
        plan.inverse_complex_inplace(&mut out);
        out.mapv(|c| c.re)
    }

    fn forward_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
        output: &mut Array2<Self::Spectrum>,
    ) {
        fill_complex32(input, output);
        plan.forward_complex_inplace(output);
    }

    fn inverse_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self::Spectrum>,
        output: &mut Array2<Self>,
        scratch: &mut Array2<Self::Spectrum>,
    ) {
        scratch.assign(input);
        plan.inverse_complex_inplace(scratch);
        fill_real32(scratch, output);
    }

    fn forward_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
    ) -> Array3<Self::Spectrum> {
        let mut out = input.mapv(|v| Complex32::new(v, 0.0));
        plan.forward_complex_inplace(&mut out);
        out
    }

    fn inverse_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self::Spectrum>,
    ) -> Array3<Self> {
        let mut out = input.clone();
        plan.inverse_complex_inplace(&mut out);
        out.mapv(|c| c.re)
    }

    fn forward_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
        output: &mut Array3<Self::Spectrum>,
    ) {
        fill_complex32(input, output);
        plan.forward_complex_inplace(output);
    }

    fn inverse_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self::Spectrum>,
        output: &mut Array3<Self>,
        scratch: &mut Array3<Self::Spectrum>,
    ) {
        scratch.assign(input);
        plan.inverse_complex_inplace(scratch);
        fill_real32(scratch, output);
    }
}

impl RealFftData for f16 {
    type Spectrum = Complex32;
    type PlanScalar = f32;

    fn forward_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
    ) -> Array1<Self::Spectrum> {
        let mut output = input.mapv(|v| Complex32::new(v.to_f32(), 0.0));
        plan.forward_complex_inplace(&mut output);
        output
    }

    fn inverse_1d(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self::Spectrum>,
    ) -> Array1<Self> {
        let mut scratch = input.clone();
        plan.inverse_complex_inplace(&mut scratch);
        scratch.mapv(|c| f16::from_f32(c.re))
    }

    fn forward_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self>,
        output: &mut Array1<Self::Spectrum>,
    ) {
        fill_complex32_from_f16(input, output);
        plan.forward_complex_inplace(output);
    }

    fn inverse_1d_into(
        plan: &FftPlan1D<Self::PlanScalar>,
        input: &Array1<Self::Spectrum>,
        output: &mut Array1<Self>,
        scratch: &mut Array1<Self::Spectrum>,
    ) {
        scratch.assign(input);
        plan.inverse_complex_inplace(scratch);
        fill_f16_from_complex32(scratch, output);
    }

    fn forward_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
    ) -> Array2<Self::Spectrum> {
        let mut out = input.mapv(|v| Complex32::new(v.to_f32(), 0.0));
        plan.forward_complex_inplace(&mut out);
        out
    }

    fn inverse_2d(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self::Spectrum>,
    ) -> Array2<Self> {
        let mut out = input.clone();
        plan.inverse_complex_inplace(&mut out);
        out.mapv(|c| f16::from_f32(c.re))
    }

    fn forward_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self>,
        output: &mut Array2<Self::Spectrum>,
    ) {
        fill_complex32_from_f16(input, output);
        plan.forward_complex_inplace(output);
    }

    fn inverse_2d_into(
        plan: &FftPlan2D<Self::PlanScalar>,
        input: &Array2<Self::Spectrum>,
        output: &mut Array2<Self>,
        scratch: &mut Array2<Self::Spectrum>,
    ) {
        scratch.assign(input);
        plan.inverse_complex_inplace(scratch);
        fill_f16_from_complex32(scratch, output);
    }

    fn forward_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
    ) -> Array3<Self::Spectrum> {
        let mut out = input.mapv(|v| Complex32::new(v.to_f32(), 0.0));
        plan.forward_complex_inplace(&mut out);
        out
    }

    fn inverse_3d(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self::Spectrum>,
    ) -> Array3<Self> {
        let mut out = input.clone();
        plan.inverse_complex_inplace(&mut out);
        out.mapv(|c| f16::from_f32(c.re))
    }

    fn forward_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self>,
        output: &mut Array3<Self::Spectrum>,
    ) {
        fill_complex32_from_f16(input, output);
        plan.forward_complex_inplace(output);
    }

    fn inverse_3d_into(
        plan: &FftPlan3D<Self::PlanScalar>,
        input: &Array3<Self::Spectrum>,
        output: &mut Array3<Self>,
        scratch: &mut Array3<Self::Spectrum>,
    ) {
        scratch.assign(input);
        plan.inverse_complex_inplace(scratch);
        fill_f16_from_complex32(scratch, output);
    }
}
