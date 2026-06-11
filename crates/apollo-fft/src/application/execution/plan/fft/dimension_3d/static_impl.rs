use super::GATHER_TILE;
use super::MOIRAI_PARALLEL_THRESHOLD;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_3d_x_scratch, with_3d_y_scratch, PlanScratch,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::dimension_1d::StaticFftPlan1D;
use core::marker::PhantomData;
use leto::ArrayViewMut3;
use ndarray::Array3;
use num_complex::Complex;

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
        assert_eq!(data.dim(), (NX, NY, NZ), "static 3D forward shape mismatch");
        let view = ArrayViewMut3::from(data.view_mut());
        self.forward_complex_leto_inplace(view);
    }

    /// Inverse transform of a complex field in-place with normalization.
    #[inline]
    pub fn inverse_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(data.dim(), (NX, NY, NZ), "static 3D inverse shape mismatch");
        let view = ArrayViewMut3::from(data.view_mut());
        self.inverse_complex_leto_inplace(view);
    }

    /// Forward transform of a complex Leto view in-place.
    #[inline]
    pub fn forward_complex_leto_inplace(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [NX, NY, NZ],
            "static 3D forward shape mismatch"
        );
        Self::axis2_pass_complex::<true>(data.reborrow());
        Self::axis1_pass_complex::<true>(data.reborrow());
        Self::axis0_pass_complex::<true>(data);
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    #[inline]
    pub fn inverse_complex_leto_inplace(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [NX, NY, NZ],
            "static 3D inverse shape mismatch"
        );
        Self::axis0_pass_complex::<false>(data.reborrow());
        Self::axis1_pass_complex::<false>(data.reborrow());
        Self::axis2_pass_complex::<false>(data);
    }

    fn axis2_pass_complex<const FORWARD: bool>(mut data: ArrayViewMut3<'_, F::Complex>) {
        if NZ <= 1 {
            return;
        }
        let data_slice = data
            .as_mut_slice_memory_order()
            .expect("3D complex data must be contiguous");
        let lane_plan = StaticFftPlan1D::<F, NZ>::new();
        let lane_fn = |lane: &mut [F::Complex]| {
            if FORWARD {
                lane_plan.forward_complex_slice_inplace(lane);
            } else {
                lane_plan.inverse_complex_slice_inplace(lane);
            }
        };
        moirai::for_each_chunk_mut_with::<
            moirai::AdaptiveWithThreshold<MOIRAI_PARALLEL_THRESHOLD>,
            _,
            _,
        >(data_slice, NZ, lane_fn);
    }

    fn axis1_pass_complex<const FORWARD: bool>(mut data: ArrayViewMut3<'_, F::Complex>) {
        if NY <= 1 {
            return;
        }
        let data_slice = data
            .as_mut_slice_memory_order()
            .expect("3D complex data must be contiguous");
        with_3d_y_scratch::<F::Complex, _>(NX * NY * NZ, |scratch| {
            for i in 0..NX {
                for j_t in (0..NY).step_by(GATHER_TILE) {
                    let j_end = (j_t + GATHER_TILE).min(NY);
                    for k_t in (0..NZ).step_by(GATHER_TILE) {
                        let k_end = (k_t + GATHER_TILE).min(NZ);
                        for j in j_t..j_end {
                            let src = (i * NY + j) * NZ;
                            for k in k_t..k_end {
                                scratch[(i * NZ + k) * NY + j] = data_slice[src + k];
                            }
                        }
                    }
                }
            }

            let lane_plan = StaticFftPlan1D::<F, NY>::new();
            let lane_fn = |lane: &mut [F::Complex]| {
                if FORWARD {
                    lane_plan.forward_complex_slice_inplace(lane);
                } else {
                    lane_plan.inverse_complex_slice_inplace(lane);
                }
            };
            moirai::for_each_chunk_mut_with::<
                moirai::AdaptiveWithThreshold<MOIRAI_PARALLEL_THRESHOLD>,
                _,
                _,
            >(scratch, NY, lane_fn);

            for i in 0..NX {
                for j_t in (0..NY).step_by(GATHER_TILE) {
                    let j_end = (j_t + GATHER_TILE).min(NY);
                    for k_t in (0..NZ).step_by(GATHER_TILE) {
                        let k_end = (k_t + GATHER_TILE).min(NZ);
                        for j in j_t..j_end {
                            let dst = (i * NY + j) * NZ;
                            for k in k_t..k_end {
                                data_slice[dst + k] = scratch[(i * NZ + k) * NY + j];
                            }
                        }
                    }
                }
            }
        });
    }

    fn axis0_pass_complex<const FORWARD: bool>(mut data: ArrayViewMut3<'_, F::Complex>) {
        if NX <= 1 {
            return;
        }
        let data_slice = data
            .as_mut_slice_memory_order()
            .expect("3D complex data must be contiguous");
        with_3d_x_scratch::<F::Complex, _>(NX * NY * NZ, |scratch| {
            for i in 0..NX {
                let src_base = i * NY * NZ;
                for j_t in (0..NY).step_by(GATHER_TILE) {
                    let j_end = (j_t + GATHER_TILE).min(NY);
                    for k_t in (0..NZ).step_by(GATHER_TILE) {
                        let k_end = (k_t + GATHER_TILE).min(NZ);
                        for j in j_t..j_end {
                            let src = src_base + j * NZ;
                            for k in k_t..k_end {
                                scratch[(j * NZ + k) * NX + i] = data_slice[src + k];
                            }
                        }
                    }
                }
            }

            let lane_plan = StaticFftPlan1D::<F, NX>::new();
            let lane_fn = |lane: &mut [F::Complex]| {
                if FORWARD {
                    lane_plan.forward_complex_slice_inplace(lane);
                } else {
                    lane_plan.inverse_complex_slice_inplace(lane);
                }
            };
            moirai::for_each_chunk_mut_with::<
                moirai::AdaptiveWithThreshold<MOIRAI_PARALLEL_THRESHOLD>,
                _,
                _,
            >(scratch, NX, lane_fn);

            for i in 0..NX {
                let dst_base = i * NY * NZ;
                for j_t in (0..NY).step_by(GATHER_TILE) {
                    let j_end = (j_t + GATHER_TILE).min(NY);
                    for k_t in (0..NZ).step_by(GATHER_TILE) {
                        let k_end = (k_t + GATHER_TILE).min(NZ);
                        for j in j_t..j_end {
                            let dst = dst_base + j * NZ;
                            for k in k_t..k_end {
                                data_slice[dst + k] = scratch[(j * NZ + k) * NX + i];
                            }
                        }
                    }
                }
            }
        });
    }
}
