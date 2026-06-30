use super::helpers::cached_power_of_two_twiddle;
use super::GATHER_TILE;
use super::MOIRAI_PARALLEL_THRESHOLD;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_3d_x_scratch, with_3d_y_scratch, PlanScratch,
};
use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use crate::domain::metadata::shape::Shape3D;
use leto::ArrayViewMut3;
use leto::Array3;
use eunomia::Complex;
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
        let (nx, ny, nz) = (shape.nx, shape.ny, shape.nz);
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
        Shape3D {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
        }
    }

    /// Forward transform of a complex field in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.shape(),
            (self.nx, self.ny, self.nz),
            "complex forward shape mismatch"
        );
        let view = ArrayViewMut3::from(data.view_mut());
        self.forward_complex_leto_inplace(view);
    }

    /// Inverse transform of a complex field in-place with FFTW-compatible normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.shape(),
            (self.nx, self.ny, self.nz),
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
            (self.nx, self.ny, self.nz),
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
            (self.nx, self.ny, self.nz),
            "axis FFT shape mismatch"
        );
        assert!(axis < 3, "axis must be 0, 1, or 2");
        self.axis_pass_complex::<false>(ArrayViewMut3::from(data.view_mut()), axis);
    }

    /// Forward transform of a complex Leto view in-place.
    pub fn forward_complex_leto_inplace(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex forward shape mismatch"
        );
        self.axis_pass_complex::<true>(data.reborrow(), 2);
        self.axis_pass_complex::<true>(data.reborrow(), 1);
        self.axis_pass_complex::<true>(data, 0);
    }

    /// Inverse transform of a complex Leto view in-place with FFTW-compatible normalization.
    pub fn inverse_complex_leto_inplace(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex inverse shape mismatch"
        );
        self.axis_pass_complex::<false>(data.reborrow(), 0);
        self.axis_pass_complex::<false>(data.reborrow(), 1);
        self.axis_pass_complex::<false>(data, 2);
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
            .as_mut_slice_memory_order()
            .expect("3D complex data must be contiguous");
        with_3d_y_scratch::<F::Complex, _>(self.nx * self.ny * self.nz, |scratch| {
            for i in 0..self.nx {
                for j_t in (0..self.ny).step_by(GATHER_TILE) {
                    let j_end = (j_t + GATHER_TILE).min(self.ny);
                    for k_t in (0..self.nz).step_by(GATHER_TILE) {
                        let k_end = (k_t + GATHER_TILE).min(self.nz);
                        for j in j_t..j_end {
                            let src = (i * self.ny + j) * self.nz;
                            for k in k_t..k_end {
                                scratch[(i * self.nz + k) * self.ny + j] = data_slice[src + k];
                            }
                        }
                    }
                }
            }
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
            for i in 0..self.nx {
                for j_t in (0..self.ny).step_by(GATHER_TILE) {
                    let j_end = (j_t + GATHER_TILE).min(self.ny);
                    for k_t in (0..self.nz).step_by(GATHER_TILE) {
                        let k_end = (k_t + GATHER_TILE).min(self.nz);
                        for j in j_t..j_end {
                            let dst = (i * self.ny + j) * self.nz;
                            for k in k_t..k_end {
                                data_slice[dst + k] = scratch[(i * self.nz + k) * self.ny + j];
                            }
                        }
                    }
                }
            }
        });
    }

    fn axis0_pass_complex<const FORWARD: bool>(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice_memory_order()
            .expect("3D complex data must be contiguous");
        with_3d_x_scratch::<F::Complex, _>(self.nx * self.ny * self.nz, |scratch| {
            for i in 0..self.nx {
                let src_base = i * self.ny * self.nz;
                for j_t in (0..self.ny).step_by(GATHER_TILE) {
                    let j_end = (j_t + GATHER_TILE).min(self.ny);
                    for k_t in (0..self.nz).step_by(GATHER_TILE) {
                        let k_end = (k_t + GATHER_TILE).min(self.nz);
                        for j in j_t..j_end {
                            let src = src_base + j * self.nz;
                            for k in k_t..k_end {
                                scratch[(j * self.nz + k) * self.nx + i] = data_slice[src + k];
                            }
                        }
                    }
                }
            }
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
            for i in 0..self.nx {
                let dst_base = i * self.ny * self.nz;
                for j_t in (0..self.ny).step_by(GATHER_TILE) {
                    let j_end = (j_t + GATHER_TILE).min(self.ny);
                    for k_t in (0..self.nz).step_by(GATHER_TILE) {
                        let k_end = (k_t + GATHER_TILE).min(self.nz);
                        for j in j_t..j_end {
                            let dst = dst_base + j * self.nz;
                            for k in k_t..k_end {
                                data_slice[dst + k] = scratch[(j * self.nz + k) * self.nx + i];
                            }
                        }
                    }
                }
            }
        });
    }

    fn axis2_pass_complex<const FORWARD: bool>(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        if self.nz <= 1 {
            return;
        }
        let data_slice = data
            .as_mut_slice_memory_order()
            .expect("3D complex data must be contiguous");
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
