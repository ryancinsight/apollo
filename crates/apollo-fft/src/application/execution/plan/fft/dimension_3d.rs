#![allow(clippy::uninit_vec)]
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
//! X[kx,ky,kz] = sum x[x,y,z] * exp(-2*pi*i*(kx*x/nx + ky*y/ny + kz*z/nz))
//!
//! and the inverse transform is
//!
//! x[x,y,z] = (1/(nx*ny*nz)) * sum X[kx,ky,kz] * exp(2*pi*i*(kx*x/nx+ky*y/ny+kz*z/nz))
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
use crate::application::execution::plan::fft::workspace::{
    uninit_copy_vec, UninitWorkspaceElement,
};
use crate::domain::metadata::shape::Shape3D;
use ndarray::{Array3, Axis};
use num_complex::Complex;
use rayon::prelude::*;
use std::sync::Arc;

/// Use rayon parallel iteration when total elements exceed this threshold.
/// Below the threshold, sequential iteration avoids rayon task-spawn overhead
/// that dominates for small volumes (e.g. 8^3 = 512 elements).
const RAYON_THRESHOLD: usize = 32768;

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
    // --- preallocated scratch for y and x gather-FFT-scatter passes ---
    scratch_y: std::sync::Mutex<Vec<F::Complex>>,
    scratch_x: std::sync::Mutex<Vec<F::Complex>>,
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
    Complex<F>: UninitWorkspaceElement,
{
    /// Create a new 3D plan.
    #[must_use]
    pub fn new(shape: Shape3D) -> Self {
        let (nx, ny, nz) = (shape.nx, shape.ny, shape.nz);
        let vol = nx * ny * nz;
        let make = |n: usize, forward: bool| -> Option<Arc<[F::Complex]>> {
            if n > 1 && n.is_power_of_two() {
                Some(if forward {
                    F::cached_twiddle_fwd(n)
                } else {
                    F::cached_twiddle_inv(n)
                })
            } else {
                None
            }
        };
        // Half-length z-axis sub-FFT for r2c/c2r (m = nz/2).
        let m = nz / 2;
        let nz_c_val = m + 1; // = nz/2+1
        Self {
            nx,
            ny,
            nz,
            nz_c: nz_c_val,
            twiddle_z_fwd: make(nz, true),
            twiddle_z_inv: make(nz, false),
            twiddle_y_fwd: make(ny, true),
            twiddle_y_inv: make(ny, false),
            twiddle_x_fwd: make(nx, true),
            twiddle_x_inv: make(nx, false),
            scratch_y: std::sync::Mutex::new(uninit_copy_vec(vol)),
            scratch_x: std::sync::Mutex::new(uninit_copy_vec(vol)),
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
        self.axis_pass_complex(data, Axis(2), true);
        self.axis_pass_complex(data, Axis(1), true);
        self.axis_pass_complex(data, Axis(0), true);
    }

    /// Inverse transform of a complex field in-place with FFTW-compatible normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.dim(),
            (self.nx, self.ny, self.nz),
            "complex inverse shape mismatch"
        );
        self.axis_pass_complex(data, Axis(0), false);
        self.axis_pass_complex(data, Axis(1), false);
        self.axis_pass_complex(data, Axis(2), false);
    }

    /// Forward complex FFT along a single `axis` (0, 1, or 2) in-place.
    ///
    /// The batched, cache-tiled per-axis building block of
    /// [`Self::forward_complex_inplace`] — it transforms all pencils along `axis`
    /// at once (32×32 tiled gather/scatter for non-contiguous axes). Exposing it
    /// lets callers that need only one axis (e.g. spectral derivatives `∂/∂xₐ`)
    /// avoid a full 3-D transform. Unnormalized (1-D forward convention); an axis
    /// of extent 1 is a no-op.
    ///
    /// # Panics
    /// - Shape mismatch with the plan, or `axis >= 3`.
    pub fn forward_axis_complex_inplace(&self, data: &mut Array3<F::Complex>, axis: usize) {
        assert_eq!(
            data.dim(),
            (self.nx, self.ny, self.nz),
            "axis FFT shape mismatch"
        );
        assert!(axis < 3, "axis must be 0, 1, or 2");
        self.axis_pass_complex(data, Axis(axis), true);
    }

    /// Inverse complex FFT along a single `axis` in-place, normalized by that
    /// axis's length, so `forward_axis` then `inverse_axis` along the same axis is
    /// the identity. See [`Self::forward_axis_complex_inplace`].
    ///
    /// # Panics
    /// - Shape mismatch with the plan, or `axis >= 3`.
    pub fn inverse_axis_complex_inplace(&self, data: &mut Array3<F::Complex>, axis: usize) {
        assert_eq!(
            data.dim(),
            (self.nx, self.ny, self.nz),
            "axis FFT shape mismatch"
        );
        assert!(axis < 3, "axis must be 0, 1, or 2");
        self.axis_pass_complex(data, Axis(axis), false);
    }

    fn axis_pass_complex(&self, data: &mut Array3<F::Complex>, axis: Axis, forward: bool) {
        if data.len_of(axis) <= 1 {
            return;
        }
        if axis.index() == 2 {
            self.axis2_pass_complex(data, forward);
            return;
        }
        if axis.index() == 1 {
            self.axis1_pass_complex(data, forward);
            return;
        }
        if axis.index() == 0 {
            self.axis0_pass_complex(data, forward);
        }
    }

    fn axis1_pass_complex(&self, data: &mut Array3<F::Complex>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        let mut scratch = self.scratch_y.lock().expect("scratch_y mutex poisoned");
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
        let lane_fn =
            |lane: &mut [F::Complex]| match (forward, &self.twiddle_y_fwd, &self.twiddle_y_inv) {
                (true, Some(tw), _) => dispatch_inplace::<F, false, false>(lane, Some(tw.as_ref())),
                (false, _, Some(tw)) => dispatch_inplace::<F, true, true>(lane, Some(tw.as_ref())),
                _ => {
                    if forward {
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
        if scratch.len() > RAYON_THRESHOLD {
            scratch.par_chunks_mut(self.ny).for_each(lane_fn);
        } else {
            scratch.chunks_mut(self.ny).for_each(lane_fn);
        }
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
    }

    fn axis0_pass_complex(&self, data: &mut Array3<F::Complex>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        let mut scratch = self.scratch_x.lock().expect("scratch_x mutex poisoned");
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
        let lane_fn =
            |lane: &mut [F::Complex]| match (forward, &self.twiddle_x_fwd, &self.twiddle_x_inv) {
                (true, Some(tw), _) => dispatch_inplace::<F, false, false>(lane, Some(tw.as_ref())),
                (false, _, Some(tw)) => dispatch_inplace::<F, true, true>(lane, Some(tw.as_ref())),
                _ => {
                    if forward {
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
        if scratch.len() > RAYON_THRESHOLD {
            scratch.par_chunks_mut(self.nx).for_each(lane_fn);
        } else {
            scratch.chunks_mut(self.nx).for_each(lane_fn);
        }
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
    }

    fn axis2_pass_complex(&self, data: &mut Array3<F::Complex>, forward: bool) {
        if self.nz <= 1 {
            return;
        }
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        let lane_fn =
            |lane: &mut [F::Complex]| match (forward, &self.twiddle_z_fwd, &self.twiddle_z_inv) {
                (true, Some(tw), _) => dispatch_inplace::<F, false, false>(lane, Some(tw.as_ref())),
                (false, _, Some(tw)) => dispatch_inplace::<F, true, true>(lane, Some(tw.as_ref())),
                _ => {
                    if forward {
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
        if data_slice.len() > RAYON_THRESHOLD {
            data_slice.par_chunks_mut(self.nz).for_each(lane_fn);
        } else {
            data_slice.chunks_mut(self.nz).for_each(lane_fn);
        }
    }
}

#[cfg(test)]
mod axis_pass_tests {
    use super::FftPlan3D;
    use crate::domain::metadata::shape::Shape3D;
    use ndarray::Array3;
    use num_complex::Complex64;

    #[test]
    fn axis_passes_compose_to_full_forward_and_roundtrip_per_axis() {
        let (nx, ny, nz) = (6usize, 4usize, 8usize);
        let plan = FftPlan3D::<f64>::new(Shape3D { nx, ny, nz });
        let original = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            let x = ((i * ny + j) * nz + k) as f64;
            Complex64::new((0.17 * x).sin() + 0.3, 0.23 * (0.31 * x).cos())
        });

        // Sequential per-axis forward (z, y, x) equals the full separable forward.
        let mut full = original.clone();
        plan.forward_complex_inplace(&mut full);
        let mut composed = original.clone();
        plan.forward_axis_complex_inplace(&mut composed, 2);
        plan.forward_axis_complex_inplace(&mut composed, 1);
        plan.forward_axis_complex_inplace(&mut composed, 0);
        let err = composed
            .iter()
            .zip(full.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(err <= 1.0e-10, "axis compose != full forward, err={err:.2e}");

        // forward_axis then inverse_axis along the same axis is the identity.
        for axis in 0..3 {
            let mut d = original.clone();
            plan.forward_axis_complex_inplace(&mut d, axis);
            plan.inverse_axis_complex_inplace(&mut d, axis);
            let err = d
                .iter()
                .zip(original.iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0_f64, f64::max);
            assert!(err <= 1.0e-10, "axis {axis} roundtrip not identity, err={err:.2e}");
        }
    }
}

impl FftPlan3D<f64> {
    /// Packed-real forward FFT along the **contiguous z-axis**, producing the
    /// half-spectrum `(nx, ny, nz/2+1)` of a real field.
    ///
    /// Instead of a full length-`nz` complex FFT followed by truncation, this
    /// packs each real z-pencil (`nz` reals) into `nz/2` complex
    /// (`h[t] = x[2t] + i·x[2t+1]`), runs a length-`nz/2` complex FFT, and unpacks
    /// the real spectrum by Hermitian symmetry — ~half the z-axis FFT work and
    /// half the z scratch. The y/x axes are **not** transformed (the caller
    /// composes them with [`Self::forward_axis_complex_inplace`]).
    ///
    /// `packed_scratch` must be `(nx, ny, nz/2)` and is used as the packed
    /// working buffer (supply a reused buffer to avoid per-call allocation).
    /// Requires even `nz`.
    ///
    /// # Panics
    /// - Shape mismatch, or odd `nz`.
    pub fn forward_real_z_into(
        &self,
        real: &Array3<f64>,
        half_out: &mut Array3<num_complex::Complex64>,
        packed_scratch: &mut Array3<num_complex::Complex64>,
    ) {
        use num_complex::Complex64;
        let (nx, ny, nz) = self.dimensions();
        assert_eq!(real.dim(), (nx, ny, nz), "real shape mismatch");
        assert_eq!(nz % 2, 0, "forward_real_z_into requires even nz");
        let m = nz / 2;
        let nz_c = m + 1;
        assert_eq!(half_out.dim(), (nx, ny, nz_c), "half_out shape mismatch");
        assert_eq!(packed_scratch.dim(), (nx, ny, m), "packed scratch must be (nx,ny,nz/2)");

        // Pack each z-pencil: h[t] = x[2t] + i·x[2t+1].
        let packed = packed_scratch;
        for i in 0..nx {
            for j in 0..ny {
                for t in 0..m {
                    packed[[i, j, t]] = Complex64::new(real[[i, j, 2 * t]], real[[i, j, 2 * t + 1]]);
                }
            }
        }

        // Length-m complex FFT along z (cached plan).
        let m_plan = <f64 as crate::PlanCacheProvider>::get_3d_plan(crate::Shape3D { nx, ny, nz: m });
        m_plan.forward_axis_complex_inplace(packed, 2);

        // Unpack the real-FFT spectrum X[k], k = 0..m, by Hermitian symmetry:
        //   E[k] = 0.5(H[k] + conj(H[m-k])),  O[k] = -0.5i(H[k] - conj(H[m-k])),
        //   X[k] = E[k] + W_N^k O[k],  W_N = exp(-2πi/nz).
        let n_f = nz as f64;
        for i in 0..nx {
            for j in 0..ny {
                let h0 = packed[[i, j, 0]];
                half_out[[i, j, 0]] = Complex64::new(h0.re + h0.im, 0.0); // DC
                half_out[[i, j, m]] = Complex64::new(h0.re - h0.im, 0.0); // Nyquist
                for k in 1..m {
                    let hk = packed[[i, j, k]];
                    let hmk = packed[[i, j, m - k]];
                    let fe = 0.5 * (hk + hmk.conj());
                    let fo = Complex64::new(0.0, -0.5) * (hk - hmk.conj());
                    let wk = Complex64::from_polar(1.0, -std::f64::consts::TAU * (k as f64) / n_f);
                    half_out[[i, j, k]] = fe + wk * fo;
                }
            }
        }
    }
}

#[cfg(test)]
mod real_z_tests {
    use super::FftPlan3D;
    use crate::domain::metadata::shape::Shape3D;
    use ndarray::{s, Array3};
    use num_complex::Complex64;

    #[test]
    fn packed_real_z_matches_full_c2c_truncation() {
        for &(nx, ny, nz) in &[(5usize, 4usize, 8usize), (3, 1, 16), (4, 6, 6)] {
            let nz_c = nz / 2 + 1;
            let plan = FftPlan3D::<f64>::new(Shape3D { nx, ny, nz });
            let real = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                let x = ((i * 71 + j * 13 + k * 5) % 97) as f64 / 97.0 - 0.5;
                (x * 6.1).sin() + 0.2 * x + 0.3
            });

            // Packed-real z-FFT.
            let mut half_new = Array3::zeros((nx, ny, nz_c));
            let mut packed = Array3::zeros((nx, ny, nz / 2));
            plan.forward_real_z_into(&real, &mut half_new, &mut packed);

            // Reference: full c2c z-pass, truncated to nz_c.
            let mut full = real.mapv(|v| Complex64::new(v, 0.0));
            plan.forward_axis_complex_inplace(&mut full, 2);
            let ref_half = full.slice(s![.., .., 0..nz_c]).to_owned();

            let err = half_new
                .iter()
                .zip(ref_half.iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0_f64, f64::max);
            assert!(err <= 1e-10, "packed-real z ({nx},{ny},{nz}) vs full-c2c: {err:.2e}");
        }
    }
}
