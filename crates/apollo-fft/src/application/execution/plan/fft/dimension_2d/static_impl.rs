use super::super::lanes;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_2d_scratch, PlanScratch,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::dimension_1d::StaticFftPlan1D;
use crate::application::execution::plan::fft::layout::{transpose_matrices, with_c_order_view};
use core::marker::PhantomData;
use eunomia::Complex;
use leto::{Array2, ArrayViewMut2};

/// Zero-sized 2D FFT plan for compile-time-known shapes.
///
/// Both axes are encoded as const generics. Row and column lane execution uses
/// `StaticFftPlan1D`, so power-of-two and selected composite/Rader lengths
/// route through monomorphized 1D kernels without storing twiddle fields or
/// function pointers in the plan value.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticFftPlan2D<F: MixedRadixScalar, const NX: usize, const NY: usize> {
    precision: PhantomData<F>,
}

impl<F: MixedRadixScalar, const NX: usize, const NY: usize> StaticFftPlan2D<F, NX, NY> {
    /// Construct a zero-sized static 2D plan.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            precision: PhantomData,
        }
    }

    /// Return the compile-time shape.
    #[must_use]
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (NX, NY)
    }
}

impl<F, const NX: usize, const NY: usize> StaticFftPlan2D<F, NX, NY>
where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    /// Forward transform of a complex array in-place.
    #[inline]
    pub fn forward_complex_inplace(&self, data: &mut Array2<F::Complex>) {
        assert_eq!(data.shape(), [NX, NY], "static 2D forward shape mismatch");
        let view = ArrayViewMut2::from(data.view_mut());
        self.forward_complex_leto_inplace(view);
    }

    /// Inverse transform of a complex array in-place with normalization.
    #[inline]
    pub fn inverse_complex_inplace(&self, data: &mut Array2<F::Complex>) {
        assert_eq!(data.shape(), [NX, NY], "static 2D inverse shape mismatch");
        let view = ArrayViewMut2::from(data.view_mut());
        self.inverse_complex_leto_inplace(view);
    }

    /// Forward transform of a complex Leto view in-place.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    #[inline]
    pub fn forward_complex_leto_inplace(&self, data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(data.shape(), [NX, NY], "static 2D forward shape mismatch");
        with_c_order_view(data, |mut contiguous| {
            Self::axis1_pass_complex::<true>(contiguous.reborrow());
            Self::axis0_pass_complex::<true>(contiguous);
        });
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    #[inline]
    pub fn inverse_complex_leto_inplace(&self, data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(data.shape(), [NX, NY], "static 2D inverse shape mismatch");
        with_c_order_view(data, |mut contiguous| {
            Self::axis0_pass_complex::<false>(contiguous.reborrow());
            Self::axis1_pass_complex::<false>(contiguous);
        });
    }

    fn axis1_pass_complex<const FORWARD: bool>(mut data: ArrayViewMut2<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 2D axis execution receives C-order data");
        let lane_plan = StaticFftPlan1D::<F, NY>::new();
        let lane_fn = |lane: &mut [F::Complex]| {
            if FORWARD {
                lane_plan.forward_complex_slice_inplace(lane);
            } else {
                lane_plan.inverse_complex_slice_inplace(lane);
            }
        };
        lanes::contiguous::<F, FORWARD, 2>(data_slice, NY, lane_fn);
    }

    fn axis0_pass_complex<const FORWARD: bool>(mut data: ArrayViewMut2<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 2D axis execution receives C-order data");
        with_2d_scratch::<F::Complex, _>(NX * NY, |scratch| {
            transpose_matrices(data_slice, scratch, 1, NX, NY);

            let lane_plan = StaticFftPlan1D::<F, NX>::new();
            let lane_fn = |lane: &mut [F::Complex]| {
                if FORWARD {
                    lane_plan.forward_complex_slice_inplace(lane);
                } else {
                    lane_plan.inverse_complex_slice_inplace(lane);
                }
            };
            lanes::execute::<F, FORWARD>(scratch, data_slice, NX, lane_fn);

            transpose_matrices(scratch, data_slice, 1, NY, NX);
        });
    }
}
