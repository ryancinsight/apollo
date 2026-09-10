use super::passes::{self, AxisLanes};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::dimension_1d::StaticFftPlan1D;
use crate::application::execution::plan::fft::layout::with_c_order_view;
use core::marker::PhantomData;
use eunomia::Complex;
use leto::Array3;
use leto::ArrayViewMut3;

/// Zero-sized 3D FFT plan for compile-time-known shapes.
///
/// All axes are encoded as const generics. Lane execution uses
/// `StaticFftPlan1D`, so the plan stores no runtime shape, twiddle fields, or
/// function pointers while preserving the existing scratch transpose layout for
/// non-contiguous axes.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticFftPlan3D<F: MixedRadixScalar, const NX: usize, const NY: usize, const NZ: usize> {
    precision: PhantomData<F>,
}

impl<F: MixedRadixScalar, const NX: usize, const NY: usize, const NZ: usize>
    StaticFftPlan3D<F, NX, NY, NZ>
{
    /// Construct a zero-sized static 3D plan.
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
    pub const fn shape(&self) -> (usize, usize, usize) {
        (NX, NY, NZ)
    }

    /// Return the half-spectrum bookkeeping value `NZ / 2 + 1`.
    #[must_use]
    #[inline]
    pub const fn nz_c(&self) -> usize {
        NZ / 2 + 1
    }
}

impl<F, const NX: usize, const NY: usize, const NZ: usize> StaticFftPlan3D<F, NX, NY, NZ>
where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    /// Forward transform of a complex field in-place.
    #[inline]
    pub fn forward_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.shape(),
            [NX, NY, NZ],
            "static 3D forward shape mismatch"
        );
        let view = ArrayViewMut3::from(data.view_mut());
        self.forward_complex_leto_inplace(view);
    }

    /// Inverse transform of a complex field in-place with normalization.
    #[inline]
    pub fn inverse_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.shape(),
            [NX, NY, NZ],
            "static 3D inverse shape mismatch"
        );
        let view = ArrayViewMut3::from(data.view_mut());
        self.inverse_complex_leto_inplace(view);
    }

    /// Forward transform of a complex Leto view in-place.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    #[inline]
    pub fn forward_complex_leto_inplace(&self, data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [NX, NY, NZ],
            "static 3D forward shape mismatch"
        );
        with_c_order_view(data, |contiguous| Self::all_axes::<true>(contiguous));
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    #[inline]
    pub fn inverse_complex_leto_inplace(&self, data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [NX, NY, NZ],
            "static 3D inverse shape mismatch"
        );
        with_c_order_view(data, |contiguous| Self::all_axes::<false>(contiguous));
    }

    /// One direction's lane transform for a compile-time axis length.
    fn lane<const FORWARD: bool, const N: usize>() -> impl Fn(&mut [F::Complex]) + Send + Sync {
        let lane_plan = StaticFftPlan1D::<F, N>::new();
        move |lane: &mut [F::Complex]| {
            if FORWARD {
                lane_plan.forward_complex_slice_inplace(lane);
            } else {
                lane_plan.inverse_complex_slice_inplace(lane);
            }
        }
    }

    fn all_axes<const FORWARD: bool>(mut data: ArrayViewMut3<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 3D axis execution receives C-order data");
        passes::all_axes::<F, FORWARD, _, _, _>(
            data_slice,
            [NX, NY, NZ],
            AxisLanes {
                x: Self::lane::<FORWARD, NX>(),
                y: Self::lane::<FORWARD, NY>(),
                z: Self::lane::<FORWARD, NZ>(),
            },
        );
    }
}
