//! 3D FFT plan.
//!
//! Apollo-owned 3D FFT implementation based on separable FFT passes.
//!
//! The plan keeps the public API stable while removing production dependence on
//! external FFT engines. Forward and inverse transforms are executed by
//! applying the auto-selected 1D FFT kernel along each axis in sequence. The
//! inverse path uses FFTW-compatible normalization by dividing by the total
//! volume.
//!
//! # Mathematical contract
//!
//! For a complex field x in C^(nx x ny x nz), the forward transform is the
//! separable 3D DFT
//!
//! `X[kx,ky,kz] = sum x[x,y,z] * exp(-2*pi*i*(kx*x/nx + ky*y/ny + kz*z/nz))`
//!
//! and the inverse transform is
//!
//! `x[x,y,z] = (1/(nx*ny*nz)) * sum X[kx,ky,kz] * exp(2*pi*i*(kx*x/nx+ky*y/ny+kz*z/nz))`
//!
//! Because the transform is separable, the implementation applies the 1D FFT
//! kernel independently along each axis. This preserves linearity and the
//! expected roundtrip identity in exact arithmetic.
//!
//! # Complexity
//!
//! Let `C(n)` be the selected 1D FFT cost. The plan costs
//! `O(n_y n_z C(n_x) + n_x n_z C(n_y) + n_x n_y C(n_z))`, with
//! `C(n) = O(n log n)` for both radix-2 and Rader plan paths. Contiguous
//! innermost-axis passes mutate depth chunks in place, while non-contiguous
//! passes gather lanes into scratch buffers before scattering them back.

use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use crate::application::execution::plan::fft::dimension_1d::StaticFftPlan1D;
use crate::application::execution::plan::fft::workspace::{
    with_3d_x_scratch, with_3d_y_scratch, PlanScratch,
};
use crate::domain::metadata::shape::Shape3D;
use core::marker::PhantomData;
use ndarray::{Array3, Axis};
use num_complex::Complex;
use std::sync::Arc;

/// Use Moirai parallel iteration when total elements exceed this threshold.
/// Below the threshold, sequential iteration avoids parallel task-spawn overhead
/// that dominates for small volumes (e.g. 8^3 = 512 elements).
const MOIRAI_PARALLEL_THRESHOLD: usize = 32768;

/// Tile size for cache-blocked gather/scatter in axis-1 and axis-0 passes.
///
/// For each i-slice in axis-1, the gather is a [ny][nz] -> [nz][ny] transpose.
/// A 32x32 tile of Complex64 = 16 KB, fitting in L1 (32-48 KB).
const GATHER_TILE: usize = 32;

/// Reusable separable 3D FFT plan generic over `MixedRadixScalar`.
pub struct FftPlan3D<F: MixedRadixScalar> {
    nx: usize,
    ny: usize,
    nz: usize,
    nz_c: usize,
    // --- precomputed twiddle tables (Some iff axis length is power-of-two > 1) ---
    twiddle_z_fwd: Option<Arc<[F::Complex]>>,
    twiddle_z_inv: Option<Arc<[F::Complex]>>,
    twiddle_y_fwd: Option<Arc<[F::Complex]>>,
    twiddle_y_inv: Option<Arc<[F::Complex]>>,
    twiddle_x_fwd: Option<Arc<[F::Complex]>>,
    twiddle_x_inv: Option<Arc<[F::Complex]>>,
}

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
        Self::axis2_pass_complex::<true>(data);
        Self::axis1_pass_complex::<true>(data);
        Self::axis0_pass_complex::<true>(data);
    }

    /// Inverse transform of a complex field in-place with normalization.
    #[inline]
    pub fn inverse_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(data.dim(), (NX, NY, NZ), "static 3D inverse shape mismatch");
        Self::axis0_pass_complex::<false>(data);
        Self::axis1_pass_complex::<false>(data);
        Self::axis2_pass_complex::<false>(data);
    }

    fn axis2_pass_complex<const FORWARD: bool>(data: &mut Array3<F::Complex>) {
        if NZ <= 1 {
            return;
        }
        let data_slice = data
            .as_slice_memory_order_mut()
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

    fn axis1_pass_complex<const FORWARD: bool>(data: &mut Array3<F::Complex>) {
        if NY <= 1 {
            return;
        }
        let data_slice = data
            .as_slice_memory_order_mut()
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

    fn axis0_pass_complex<const FORWARD: bool>(data: &mut Array3<F::Complex>) {
        if NX <= 1 {
            return;
        }
        let data_slice = data
            .as_slice_memory_order_mut()
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
        // Half-length z-axis sub-FFT for r2c/c2r (m = nz/2).
        let m = nz / 2;
        let nz_c_val = m + 1; // = nz/2+1
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
            data.dim(),
            (self.nx, self.ny, self.nz),
            "complex forward shape mismatch"
        );
        self.axis_pass_complex::<true>(data, Axis(2));
        self.axis_pass_complex::<true>(data, Axis(1));
        self.axis_pass_complex::<true>(data, Axis(0));
    }

    /// Inverse transform of a complex field in-place with FFTW-compatible normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.dim(),
            (self.nx, self.ny, self.nz),
            "complex inverse shape mismatch"
        );
        self.axis_pass_complex::<false>(data, Axis(0));
        self.axis_pass_complex::<false>(data, Axis(1));
        self.axis_pass_complex::<false>(data, Axis(2));
    }

    fn axis_pass_complex<const FORWARD: bool>(&self, data: &mut Array3<F::Complex>, axis: Axis) {
        if data.len_of(axis) <= 1 {
            return;
        }
        if axis.index() == 2 {
            self.axis2_pass_complex::<FORWARD>(data);
            return;
        }
        if axis.index() == 1 {
            self.axis1_pass_complex::<FORWARD>(data);
            return;
        }
        if axis.index() == 0 {
            self.axis0_pass_complex::<FORWARD>(data);
        }
    }

    fn axis1_pass_complex<const FORWARD: bool>(&self, data: &mut Array3<F::Complex>) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        with_3d_y_scratch::<F::Complex, _>(self.nx * self.ny * self.nz, |scratch| {
            // Cache-blocked gather: data[i,j,k] (row-major) -> scratch[i,k,j].
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
            // Cache-blocked scatter: scratch[i,k,j] -> data[i,j,k].
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

    fn axis0_pass_complex<const FORWARD: bool>(&self, data: &mut Array3<F::Complex>) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        with_3d_x_scratch::<F::Complex, _>(self.nx * self.ny * self.nz, |scratch| {
            // Cache-blocked gather: data[i,j,k] -> scratch[j,k,i].
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
            // Cache-blocked scatter: scratch[j,k,i] -> data[i,j,k].
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

    fn axis2_pass_complex<const FORWARD: bool>(&self, data: &mut Array3<F::Complex>) {
        if self.nz <= 1 {
            return;
        }
        let data_slice = data
            .as_slice_memory_order_mut()
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
    use num_complex::Complex64;
    use std::f64::consts::PI;

    fn signal<const NX: usize, const NY: usize, const NZ: usize>() -> Array3<Complex64> {
        Array3::from_shape_fn((NX, NY, NZ), |(i, j, k)| {
            let x = ((i * NY + j) * NZ + k) as f64;
            Complex64::new(
                (0.17 * x).sin() + 0.11 * (0.07 * x).cos(),
                0.23 * (0.31 * x).cos(),
            )
        })
    }

    fn direct_forward<const NX: usize, const NY: usize, const NZ: usize>(
        input: &Array3<Complex64>,
    ) -> Array3<Complex64> {
        let mut out = Array3::from_elem((NX, NY, NZ), Complex64::new(0.0, 0.0));
        for kx in 0..NX {
            for ky in 0..NY {
                for kz in 0..NZ {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for x in 0..NX {
                        for y in 0..NY {
                            for z in 0..NZ {
                                let phase = -2.0
                                    * PI
                                    * ((kx * x) as f64 / NX as f64
                                        + (ky * y) as f64 / NY as f64
                                        + (kz * z) as f64 / NZ as f64);
                                acc += input[(x, y, z)] * Complex64::from_polar(1.0, phase);
                            }
                        }
                    }
                    out[(kx, ky, kz)] = acc;
                }
            }
        }
        out
    }

    fn max_err(a: &Array3<Complex64>, b: &Array3<Complex64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn static_fft_3d_plan_is_zero_sized() {
        assert_eq!(std::mem::size_of::<StaticFftPlan3D<f64, 3, 4, 5>>(), 0);
        assert_eq!(StaticFftPlan3D::<f64, 3, 4, 5>::new().shape(), (3, 4, 5));
        assert_eq!(StaticFftPlan3D::<f64, 3, 4, 5>::new().nz_c(), 3);
    }

    #[test]
    fn static_fft_3d_forward_matches_direct() {
        let plan = StaticFftPlan3D::<f64, 3, 4, 5>::new();
        let input = signal::<3, 4, 5>();
        let expected = direct_forward::<3, 4, 5>(&input);
        let mut actual = input;
        plan.forward_complex_inplace(&mut actual);
        let err = max_err(&actual, &expected);
        assert!(err <= 1.0e-10, "static 3D forward mismatch err={err:.2e}");
    }

    #[test]
    fn static_fft_3d_inverse_roundtrip_recovers_input() {
        let plan = StaticFftPlan3D::<f64, 3, 4, 5>::new();
        let input = signal::<3, 4, 5>();
        let mut actual = input.clone();
        plan.forward_complex_inplace(&mut actual);
        plan.inverse_complex_inplace(&mut actual);
        let err = max_err(&actual, &input);
        assert!(err <= 1.0e-10, "static 3D roundtrip mismatch err={err:.2e}");
    }
}
