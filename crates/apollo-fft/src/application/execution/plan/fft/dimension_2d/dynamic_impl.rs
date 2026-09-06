use super::super::lanes;
use super::super::twiddles::cached_power_of_two_twiddle;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_2d_scratch, PlanScratch,
};
use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use crate::application::execution::plan::fft::layout::{transpose_matrices, with_c_order_view};
use crate::domain::metadata::shape::Shape2D;
use eunomia::Complex;
use leto::{Array2, ArrayViewMut2};
use std::sync::Arc;

/// Reusable 2D FFT plan generic over `MixedRadixScalar`.
pub struct FftPlan2D<F: MixedRadixScalar> {
    nx: usize,
    ny: usize,
    twiddle_row_fwd: Option<Arc<[F::Complex]>>,
    twiddle_row_inv: Option<Arc<[F::Complex]>>,
    twiddle_col_fwd: Option<Arc<[F::Complex]>>,
    twiddle_col_inv: Option<Arc<[F::Complex]>>,
}

impl<F: MixedRadixScalar> std::fmt::Debug for FftPlan2D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan2D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .finish()
    }
}

impl<F> FftPlan2D<F>
where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    /// Create a new 2D plan.
    #[must_use]
    pub fn new(shape: Shape2D) -> Self {
        crate::ensure_thread_local_scratch_hook_registered();
        let (nx, ny) = (shape.nx(), shape.ny());
        Self {
            nx,
            ny,
            twiddle_row_fwd: cached_power_of_two_twiddle::<F, true>(ny),
            twiddle_row_inv: cached_power_of_two_twiddle::<F, false>(ny),
            twiddle_col_fwd: cached_power_of_two_twiddle::<F, true>(nx),
            twiddle_col_inv: cached_power_of_two_twiddle::<F, false>(nx),
        }
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape2D {
        Shape2D::new(self.nx, self.ny)
            .expect("invariant: the plan was built from a validated shape")
    }

    /// Forward transform of a complex array in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array2<F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex forward shape mismatch"
        );
        let view = ArrayViewMut2::from(data.view_mut());
        self.forward_complex_leto_inplace(view);
    }

    /// Inverse transform of a complex array in-place with normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array2<F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex inverse shape mismatch"
        );
        let view = ArrayViewMut2::from(data.view_mut());
        self.inverse_complex_leto_inplace(view);
    }

    /// Forward transform of a complex Leto view in-place.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    pub fn forward_complex_leto_inplace(&self, data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex forward shape mismatch"
        );
        with_c_order_view(data, |mut contiguous| {
            self.axis_pass_complex::<true>(contiguous.reborrow(), 1);
            self.axis_pass_complex::<true>(contiguous, 0);
        });
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    pub fn inverse_complex_leto_inplace(&self, data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex inverse shape mismatch"
        );
        with_c_order_view(data, |mut contiguous| {
            self.axis_pass_complex::<false>(contiguous.reborrow(), 0);
            self.axis_pass_complex::<false>(contiguous, 1);
        });
    }

    fn axis_pass_complex<const FORWARD: bool>(
        &self,
        data: ArrayViewMut2<'_, F::Complex>,
        axis: usize,
    ) {
        if axis == 1 {
            self.axis1_pass_complex::<FORWARD>(data);
            return;
        }
        if axis == 0 {
            self.axis0_pass_complex::<FORWARD>(data);
            return;
        }

        unreachable!("2D FFT axis index must be 0 or 1");
    }

    fn axis1_pass_complex<const FORWARD: bool>(&self, mut data: ArrayViewMut2<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 2D axis execution receives C-order data");
        let lane_fn =
            |lane: &mut [F::Complex]| match (FORWARD, &self.twiddle_row_fwd, &self.twiddle_row_inv)
            {
                (true, Some(tw), _) => dispatch_inplace::<F, false, false>(lane, Some(tw.as_ref())),
                (false, _, Some(tw)) => dispatch_inplace::<F, true, true>(lane, Some(tw.as_ref())),
                _ => {
                    if FORWARD {
                        crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(
                            lane,
                        )
                    } else {
                        crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(
                            lane,
                        )
                    }
                }
            };
        lanes::contiguous::<F, FORWARD, 2>(data_slice, self.ny, lane_fn);
    }

    fn axis0_pass_complex<const FORWARD: bool>(&self, mut data: ArrayViewMut2<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 2D axis execution receives C-order data");
        with_2d_scratch::<F::Complex, _>(self.nx * self.ny, |scratch| {
            transpose_matrices(data_slice, scratch, 1, self.nx, self.ny);
            let lane_fn = |lane: &mut [F::Complex]| match (
                FORWARD,
                &self.twiddle_col_fwd,
                &self.twiddle_col_inv,
            ) {
                (true, Some(tw), _) => dispatch_inplace::<F, false, false>(lane, Some(tw.as_ref())),
                (false, _, Some(tw)) => dispatch_inplace::<F, true, true>(lane, Some(tw.as_ref())),
                _ => {
                    if FORWARD {
                        crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(
                            lane,
                        )
                    } else {
                        crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(
                            lane,
                        )
                    }
                }
            };
            lanes::execute::<F, FORWARD>(scratch, data_slice, self.nx, lane_fn);
            transpose_matrices(scratch, data_slice, 1, self.ny, self.nx);
        });
    }
}
