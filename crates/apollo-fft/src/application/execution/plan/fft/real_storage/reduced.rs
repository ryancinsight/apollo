//! `f32` real-storage implementation: native `Complex32` spectrum, `f32` plan scalar.

use super::RealFftData;
use super::fill::{fill_complex32, fill_real32};
use crate::application::execution::plan::fft::dimension_1d::FftPlan1D;
use crate::application::execution::plan::fft::dimension_2d::FftPlan2D;
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex32;

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
