//! 2D FFT plan.
//!
//! Apollo-owned 2D FFT implementation.
//!
//! The 2D DFT is separable, so this plan applies the in-repo auto-selected 1D
//! FFT kernel across rows and columns. The inverse path is normalized on each
//! inverse axis pass, which gives the standard `1 / (nx * ny)` inverse
//! normalization.
//!
//! # Mathematical contract
//!
//! For a complex input field `x in C^(nx x ny)`, the forward transform is
//!
//! `X[k,l] = sum_i sum_j x[i,j] exp(-2*pi*i*(k*i/nx + l*j/ny))`.
//!
//! The inverse transform is
//!
//! `x[i,j] = (1/(nx*ny)) sum_k sum_l X[k,l] exp(2*pi*i*(k*i/nx + l*j/ny))`.
//!
//! The implementation is linear and separable. Floating-point error follows
//! from the selected scalar precision and the selected 1D FFT kernel.
//!
//! # Complexity
//!
//! Let `C(n)` be the selected 1D FFT cost. The plan costs
//! `O(ny * C(nx) + nx * C(ny))`, with `C(n) = O(n log n)` for both radix-2 and
//! radix-2, mixed-radix, and Rader plan paths. Contiguous innermost-axis passes mutate row chunks in
//! place, while non-contiguous passes gather lanes into scratch buffers before
//! scattering them back.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_2d_scratch, PlanScratch,
};
use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use crate::application::execution::plan::fft::dimension_1d::StaticFftPlan1D;
use crate::domain::metadata::shape::Shape2D;
use core::marker::PhantomData;
use eunomia::Complex;
use leto::Array2;
use leto::ArrayViewMut2;
use std::sync::Arc;

/// Use Moirai parallel iteration when total elements exceed this threshold.
/// Below the threshold, sequential iteration avoids parallel task-spawn overhead
/// that dominates for small matrices (e.g. 32×32 = 1024 elements).
const MOIRAI_PARALLEL_THRESHOLD: usize = 32768;

/// Tile size for cache-blocked transpose.
/// A 32×32 tile of Complex64 = 8 KB, fitting comfortably in L1 (32–48 KB).
const TRANSPOSE_TILE: usize = 32;

/// Reusable 2D FFT plan generic over `MixedRadixScalar`.
pub struct FftPlan2D<F: MixedRadixScalar> {
    nx: usize,
    ny: usize,
    twiddle_row_fwd: Option<Arc<[F::Complex]>>,
    twiddle_row_inv: Option<Arc<[F::Complex]>>,
    twiddle_col_fwd: Option<Arc<[F::Complex]>>,
    twiddle_col_inv: Option<Arc<[F::Complex]>>,
}

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
    #[inline]
    pub fn forward_complex_leto_inplace(&self, mut data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(data.shape(), [NX, NY], "static 2D forward shape mismatch");
        Self::axis1_pass_complex::<true>(data.reborrow());
        Self::axis0_pass_complex::<true>(data);
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    #[inline]
    pub fn inverse_complex_leto_inplace(&self, mut data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(data.shape(), [NX, NY], "static 2D inverse shape mismatch");
        Self::axis0_pass_complex::<false>(data.reborrow());
        Self::axis1_pass_complex::<false>(data);
    }

    fn axis1_pass_complex<const FORWARD: bool>(mut data: ArrayViewMut2<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice_memory_order()
            .expect("2D complex data must be contiguous");
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
        >(data_slice, NY, lane_fn);
    }

    fn axis0_pass_complex<const FORWARD: bool>(mut data: ArrayViewMut2<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice_memory_order()
            .expect("2D complex data must be contiguous");
        with_2d_scratch::<F::Complex, _>(NX * NY, |scratch| {
            for col_t in (0..NY).step_by(TRANSPOSE_TILE) {
                let col_end = (col_t + TRANSPOSE_TILE).min(NY);
                for row_t in (0..NX).step_by(TRANSPOSE_TILE) {
                    let row_end = (row_t + TRANSPOSE_TILE).min(NX);
                    for col in col_t..col_end {
                        for row in row_t..row_end {
                            scratch[col * NX + row] = data_slice[row * NY + col];
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

            for col_t in (0..NY).step_by(TRANSPOSE_TILE) {
                let col_end = (col_t + TRANSPOSE_TILE).min(NY);
                for row_t in (0..NX).step_by(TRANSPOSE_TILE) {
                    let row_end = (row_t + TRANSPOSE_TILE).min(NX);
                    for col in col_t..col_end {
                        for row in row_t..row_end {
                            data_slice[row * NY + col] = scratch[col * NX + row];
                        }
                    }
                }
            }
        });
    }
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
        let (nx, ny) = (shape.nx, shape.ny);
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
        Shape2D {
            nx: self.nx,
            ny: self.ny,
        }
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
    pub fn forward_complex_leto_inplace(&self, mut data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex forward shape mismatch"
        );
        self.axis_pass_complex::<true>(data.reborrow(), 1);
        self.axis_pass_complex::<true>(data, 0);
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    pub fn inverse_complex_leto_inplace(&self, mut data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex inverse shape mismatch"
        );
        self.axis_pass_complex::<false>(data.reborrow(), 0);
        self.axis_pass_complex::<false>(data, 1);
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
            .as_mut_slice_memory_order()
            .expect("2D complex data must be contiguous");
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
        moirai::for_each_chunk_mut_with::<
            moirai::AdaptiveWithThreshold<MOIRAI_PARALLEL_THRESHOLD>,
            _,
            _,
        >(data_slice, self.ny, lane_fn);
    }

    fn axis0_pass_complex<const FORWARD: bool>(&self, mut data: ArrayViewMut2<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice_memory_order()
            .expect("2D complex data must be contiguous");
        with_2d_scratch::<F::Complex, _>(self.nx * self.ny, |scratch| {
            // Cache-blocked gather: data[row, col] (row-major) → scratch[col, row] (row-major)
            // Tile size TRANSPOSE_TILE×TRANSPOSE_TILE fits in L1, avoiding cache miss
            // on the strided column-read of data.
            for col_t in (0..self.ny).step_by(TRANSPOSE_TILE) {
                let col_end = (col_t + TRANSPOSE_TILE).min(self.ny);
                for row_t in (0..self.nx).step_by(TRANSPOSE_TILE) {
                    let row_end = (row_t + TRANSPOSE_TILE).min(self.nx);
                    for col in col_t..col_end {
                        for row in row_t..row_end {
                            scratch[col * self.nx + row] = data_slice[row * self.ny + col];
                        }
                    }
                }
            }
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
            moirai::for_each_chunk_mut_with::<
                moirai::AdaptiveWithThreshold<MOIRAI_PARALLEL_THRESHOLD>,
                _,
                _,
            >(&mut scratch[..], self.nx, lane_fn);
            // Cache-blocked scatter: scratch[col, row] → data[row, col]
            for col_t in (0..self.ny).step_by(TRANSPOSE_TILE) {
                let col_end = (col_t + TRANSPOSE_TILE).min(self.ny);
                for row_t in (0..self.nx).step_by(TRANSPOSE_TILE) {
                    let row_end = (row_t + TRANSPOSE_TILE).min(self.nx);
                    for col in col_t..col_end {
                        for row in row_t..row_end {
                            data_slice[row * self.ny + col] = scratch[col * self.nx + row];
                        }
                    }
                }
            }
        });
    }
}

#[inline]
fn cached_power_of_two_twiddle<F, const FORWARD: bool>(n: usize) -> Option<Arc<[F::Complex]>>
where
    F: MixedRadixScalar,
{
    if n <= 1 || !n.is_power_of_two() {
        return None;
    }
    Some(if FORWARD {
        F::cached_twiddle_fwd(n)
    } else {
        F::cached_twiddle_inv(n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::Complex64;
    use std::f64::consts::PI;

    fn signal<const NX: usize, const NY: usize>() -> Array2<Complex64> {
        Array2::from_shape_fn([NX, NY], |[i, j]| {
            let x = (i * NY + j) as f64;
            Complex64::new(
                (0.17 * x).sin() + 0.11 * (0.07 * x).cos(),
                0.23 * (0.31 * x).cos(),
            )
        })
    }

    fn direct_forward<const NX: usize, const NY: usize>(
        input: &Array2<Complex64>,
    ) -> Array2<Complex64> {
        let mut out = Array2::from_elem([NX, NY], Complex64::new(0.0, 0.0));
        for kx in 0..NX {
            for ky in 0..NY {
                let mut acc = Complex64::new(0.0, 0.0);
                for x in 0..NX {
                    for y in 0..NY {
                        let phase =
                            -2.0 * PI * ((kx * x) as f64 / NX as f64 + (ky * y) as f64 / NY as f64);
                        acc += input[[x, y]] * Complex64::from_polar(1.0, phase);
                    }
                }
                out[[kx, ky]] = acc;
            }
        }
        out
    }

    fn max_err(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn static_fft_2d_plan_is_zero_sized() {
        assert_eq!(std::mem::size_of::<StaticFftPlan2D<f64, 4, 5>>(), 0);
        assert_eq!(StaticFftPlan2D::<f64, 4, 5>::new().shape(), (4, 5));
    }

    #[test]
    fn static_fft_2d_forward_matches_direct() {
        let plan = StaticFftPlan2D::<f64, 4, 5>::new();
        let input = signal::<4, 5>();
        let expected = direct_forward::<4, 5>(&input);
        let mut actual = input;
        plan.forward_complex_inplace(&mut actual);
        let err = max_err(&actual, &expected);
        assert!(err <= 1.0e-10, "static 2D forward mismatch err={err:.2e}");
    }

    #[test]
    fn static_fft_2d_inverse_roundtrip_recovers_input() {
        let plan = StaticFftPlan2D::<f64, 4, 5>::new();
        let input = signal::<4, 5>();
        let mut actual = input.clone();
        plan.forward_complex_inplace(&mut actual);
        plan.inverse_complex_inplace(&mut actual);
        let err = max_err(&actual, &input);
        assert!(err <= 1.0e-10, "static 2D roundtrip mismatch err={err:.2e}");
    }
}
