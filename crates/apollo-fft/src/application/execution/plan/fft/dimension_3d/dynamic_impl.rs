use super::twiddles::cached_power_of_two_twiddle;
use super::MOIRAI_PARALLEL_THRESHOLD;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_3d_x_scratch, with_3d_y_scratch, PlanScratch,
};
use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use crate::application::execution::plan::fft::layout::{transpose_matrices, with_c_order_view};
use crate::domain::metadata::shape::Shape3D;
use eunomia::Complex;
use leto::Array3;
use leto::ArrayViewMut3;
use std::sync::Arc;

/// Reusable separable 3D FFT plan generic over `MixedRadixScalar`.
pub struct FftPlan3D<F: MixedRadixScalar> {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,
    pub(crate) nz_c: usize,
    pub(crate) twiddle_z_fwd: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_z_inv: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_y_fwd: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_y_inv: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_x_fwd: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_x_inv: Option<Arc<[F::Complex]>>,
}

impl<F: MixedRadixScalar> std::fmt::Debug for FftPlan3D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan3D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("nz_c", &self.nz_c)
            .finish()
    }
}

impl<F> FftPlan3D<F>
where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    /// Create a new 3D plan.
    #[must_use]
    pub fn new(shape: Shape3D) -> Self {
        let (nx, ny, nz) = (shape.nx(), shape.ny(), shape.nz());
        let m = nz / 2;
        let nz_c_val = m + 1;
        Self {
            nx,
            ny,
            nz,
            nz_c: nz_c_val,
            twiddle_z_fwd: cached_power_of_two_twiddle::<F, true>(nz),
            twiddle_z_inv: cached_power_of_two_twiddle::<F, false>(nz),
            twiddle_y_fwd: cached_power_of_two_twiddle::<F, true>(ny),
            twiddle_y_inv: cached_power_of_two_twiddle::<F, false>(ny),
            twiddle_x_fwd: cached_power_of_two_twiddle::<F, true>(nx),
            twiddle_x_inv: cached_power_of_two_twiddle::<F, false>(nx),
        }
    }

    /// Return the half-spectrum bookkeeping value `nz / 2 + 1`.
    #[must_use]
    pub fn nz_c(&self) -> usize {
        self.nz_c
    }

    /// Return the full real-domain shape owned by this plan.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape3D {
        Shape3D::new(self.nx, self.ny, self.nz)
            .expect("invariant: the plan was built from a validated shape")
    }

    /// Forward transform of a complex field in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex forward shape mismatch"
        );
        let view = ArrayViewMut3::from(data.view_mut());
        self.forward_complex_leto_inplace(view);
    }

    /// Inverse transform of a complex field in-place with FFTW-compatible normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex inverse shape mismatch"
        );
        let view = ArrayViewMut3::from(data.view_mut());
        self.inverse_complex_leto_inplace(view);
    }

    /// Forward complex FFT along a single `axis` (0, 1, or 2) in-place.
    ///
    /// This is the batched, cache-tiled, parallel per-axis building block of
    /// [`Self::forward_complex_inplace`] — it transforms all pencils along `axis`
    /// at once (32×32 tiled gather/scatter for non-contiguous axes, Moirai
    /// parallelism over pencils, cached power-of-two twiddles). Exposing it lets
    /// callers that need only one axis (e.g. spectral derivatives `∂/∂xₐ`) avoid
    /// the cost of a full 3-D transform. Unnormalized, matching the 1-D forward
    /// convention; an `axis` whose extent is 1 is a no-op.
    ///
    /// # Panics
    /// - Shape mismatch with the plan, or `axis >= 3`.
    pub fn forward_axis_complex_inplace(&self, data: &mut Array3<F::Complex>, axis: usize) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "axis FFT shape mismatch"
        );
        assert!(axis < 3, "axis must be 0, 1, or 2");
        self.axis_pass_complex::<true>(ArrayViewMut3::from(data.view_mut()), axis);
    }

    /// Inverse complex FFT along a single `axis` in-place, normalized by that
    /// axis's length, so `forward_axis` followed by `inverse_axis` along the same
    /// axis is the identity. See [`Self::forward_axis_complex_inplace`].
    ///
    /// # Panics
    /// - Shape mismatch with the plan, or `axis >= 3`.
    pub fn inverse_axis_complex_inplace(&self, data: &mut Array3<F::Complex>, axis: usize) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "axis FFT shape mismatch"
        );
        assert!(axis < 3, "axis must be 0, 1, or 2");
        self.axis_pass_complex::<false>(ArrayViewMut3::from(data.view_mut()), axis);
    }

    /// Forward transform of a complex Leto view in-place.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    pub fn forward_complex_leto_inplace(&self, data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex forward shape mismatch"
        );
        with_c_order_view(data, |mut contiguous| {
            self.axis_pass_complex::<true>(contiguous.reborrow(), 2);
            self.axis_pass_complex::<true>(contiguous.reborrow(), 1);
            self.axis_pass_complex::<true>(contiguous, 0);
        });
    }

    /// Inverse transform of a complex Leto view in-place with FFTW-compatible normalization.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    pub fn inverse_complex_leto_inplace(&self, data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex inverse shape mismatch"
        );
        with_c_order_view(data, |mut contiguous| {
            self.axis_pass_complex::<false>(contiguous.reborrow(), 0);
            self.axis_pass_complex::<false>(contiguous.reborrow(), 1);
            self.axis_pass_complex::<false>(contiguous, 2);
        });
    }

    fn axis_pass_complex<const FORWARD: bool>(
        &self,
        data: ArrayViewMut3<'_, F::Complex>,
        axis: usize,
    ) {
        if data.shape()[axis] <= 1 {
            return;
        }
        if axis == 2 {
            self.axis2_pass_complex::<FORWARD>(data);
            return;
        }
        if axis == 1 {
            self.axis1_pass_complex::<FORWARD>(data);
            return;
        }
        if axis == 0 {
            self.axis0_pass_complex::<FORWARD>(data);
        }
    }

    fn axis1_pass_complex<const FORWARD: bool>(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 3D axis execution receives C-order data");
        with_3d_y_scratch::<F::Complex, _>(self.nx * self.ny * self.nz, |scratch| {
            transpose_matrices(data_slice, scratch, self.nx, self.ny, self.nz);
            let lane_fn = |lane: &mut [F::Complex]| match (
                FORWARD,
                &self.twiddle_y_fwd,
                &self.twiddle_y_inv,
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
            moirai::for_each_chunk_mut_with::<
                moirai::AdaptiveWithThreshold<MOIRAI_PARALLEL_THRESHOLD>,
                _,
                _,
            >(&mut scratch[..], self.ny, lane_fn);
            transpose_matrices(scratch, data_slice, self.nx, self.nz, self.ny);
        });
    }

    fn axis0_pass_complex<const FORWARD: bool>(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 3D axis execution receives C-order data");
        with_3d_x_scratch::<F::Complex, _>(self.nx * self.ny * self.nz, |scratch| {
            transpose_matrices(data_slice, scratch, 1, self.nx, self.ny * self.nz);
            let lane_fn = |lane: &mut [F::Complex]| match (
                FORWARD,
                &self.twiddle_x_fwd,
                &self.twiddle_x_inv,
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
            moirai::for_each_chunk_mut_with::<
                moirai::AdaptiveWithThreshold<MOIRAI_PARALLEL_THRESHOLD>,
                _,
                _,
            >(&mut scratch[..], self.nx, lane_fn);
            transpose_matrices(scratch, data_slice, 1, self.ny * self.nz, self.nx);
        });
    }

    fn axis2_pass_complex<const FORWARD: bool>(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        if self.nz <= 1 {
            return;
        }
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 3D axis execution receives C-order data");
        let lane_fn =
            |lane: &mut [F::Complex]| match (FORWARD, &self.twiddle_z_fwd, &self.twiddle_z_inv) {
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
        moirai::for_each_chunk_mut_with::<
            moirai::AdaptiveWithThreshold<MOIRAI_PARALLEL_THRESHOLD>,
            _,
            _,
        >(data_slice, self.nz, lane_fn);
    }
}
