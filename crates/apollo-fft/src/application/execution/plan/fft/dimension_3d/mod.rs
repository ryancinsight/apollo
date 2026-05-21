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
//! `C(n) = O(n log n)` for both radix-2 and Bluestein plan paths. Contiguous
//! innermost-axis passes mutate depth chunks in place, while non-contiguous
//! passes gather lanes into scratch buffers before scattering them back.
//!
//! # Failure modes
//!
//! - zero dimensions are rejected by `Shape3D::new`
//! - caller-supplied buffers must match the plan dimensions
//! - non-contiguous ndarray buffers panic when a contiguous slice is required

use crate::application::execution::kernel::mixed_radix::{
    cached_twiddle_fwd_32, cached_twiddle_fwd_64, cached_twiddle_inv_32, cached_twiddle_inv_64,
};
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::application::execution::plan::fft::workspace::{
    uninit_copy_vec, UninitWorkspaceElement,
};
use crate::domain::metadata::precision::PrecisionProfile;
use crate::domain::metadata::shape::Shape3D;
use half::f16;
use ndarray::{Array3, Axis, Zip};
use num_complex::Complex32;
use num_complex::Complex64;
use std::sync::Arc;

/// Use rayon parallel iteration when total elements exceed this threshold.
/// Below the threshold, sequential iteration avoids rayon task-spawn overhead
/// that dominates for small volumes (e.g. 8^3 = 512 elements).
pub(super) const RAYON_THRESHOLD: usize = 32768;

/// Tile size for cache-blocked gather/scatter in axis-1 and axis-0 passes.
///
/// For each i-slice in axis-1, the gather is a [ny][nz] -> [nz][ny] transpose.
/// A 32x32 tile of Complex64 = 16 KB, fitting in L1 (32-48 KB).
pub(super) const GATHER_TILE: usize = 32;

/// Reusable separable 3D FFT plan with precomputed axis twiddles and scratch buffers.
pub struct FftPlan3D {
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,
    pub(super) nz_c: usize,
    pub(super) precision: PrecisionProfile,
    // --- precomputed twiddle tables (Some iff axis length is power-of-two > 1) ---
    pub(super) twiddle_z_fwd_64: Option<Arc<[Complex64]>>,
    pub(super) twiddle_z_inv_64: Option<Arc<[Complex64]>>,
    pub(super) twiddle_y_fwd_64: Option<Arc<[Complex64]>>,
    pub(super) twiddle_y_inv_64: Option<Arc<[Complex64]>>,
    pub(super) twiddle_x_fwd_64: Option<Arc<[Complex64]>>,
    pub(super) twiddle_x_inv_64: Option<Arc<[Complex64]>>,
    pub(super) twiddle_z_fwd_32: Option<Arc<[Complex32]>>,
    pub(super) twiddle_z_inv_32: Option<Arc<[Complex32]>>,
    pub(super) twiddle_y_fwd_32: Option<Arc<[Complex32]>>,
    pub(super) twiddle_y_inv_32: Option<Arc<[Complex32]>>,
    pub(super) twiddle_x_fwd_32: Option<Arc<[Complex32]>>,
    pub(super) twiddle_x_inv_32: Option<Arc<[Complex32]>>,
    // --- preallocated scratch for y and x gather-FFT-scatter passes ---
    pub(super) scratch_y_64: std::sync::Mutex<Vec<Complex64>>,
    pub(super) scratch_x_64: std::sync::Mutex<Vec<Complex64>>,
    pub(super) scratch_y_32: std::sync::Mutex<Vec<Complex32>>,
    pub(super) scratch_x_32: std::sync::Mutex<Vec<Complex32>>,
    // --- r2c/c2r: half-length z-axis twiddles (m = nz/2) ---
    //
    // The r2c forward z-axis pass applies a length-m complex DFT to packed real
    // pairs, followed by Cooley-Tukey extraction. These tables are the twiddle
    // factors for the length-m sub-FFT (precomputed when m is a power of two).
    pub(super) twiddle_zh_fwd_64: Option<Arc<[Complex64]>>,
    pub(super) twiddle_zh_inv_64: Option<Arc<[Complex64]>>,
    /// Extraction twiddles W_k = exp(-2pii*k/nz) for k = 0..nz_c-1.
    /// Used in the Cooley-Tukey r2c split step and its inverse.
    pub(super) r2c_twiddles_64: Vec<Complex64>,
    // --- preallocated scratch for r2c y and x passes (half-spectrum volume) ---
    pub(super) scratch_r2c_y_64: std::sync::Mutex<Vec<Complex64>>,
    pub(super) scratch_r2c_x_64: std::sync::Mutex<Vec<Complex64>>,
}

impl std::fmt::Debug for FftPlan3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan3D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("nz_c", &self.nz_c)
            .field("precision", &self.precision)
            .finish()
    }
}

impl FftPlan3D {
    /// Create a new 3D plan.
    #[must_use]
    pub fn new(shape: Shape3D) -> Self {
        Self::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 3D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(shape: Shape3D, precision: PrecisionProfile) -> Self {
        let (nx, ny, nz) = (shape.nx, shape.ny, shape.nz);
        let vol = nx * ny * nz;
        let make64 = |n: usize, forward: bool| -> Option<Arc<[Complex64]>> {
            if n > 1 && n.is_power_of_two() {
                Some(if forward {
                    cached_twiddle_fwd_64(n)
                } else {
                    cached_twiddle_inv_64(n)
                })
            } else {
                None
            }
        };
        let make32 = |n: usize, forward: bool| -> Option<Arc<[Complex32]>> {
            if n > 1 && n.is_power_of_two() {
                Some(if forward {
                    cached_twiddle_fwd_32(n)
                } else {
                    cached_twiddle_inv_32(n)
                })
            } else {
                None
            }
        };
        // Half-length z-axis sub-FFT for r2c/c2r (m = nz/2).
        let m = nz / 2;
        let nz_c_val = m + 1; // = nz/2+1
        let r2c_vol = nx * ny * nz_c_val;
        // Extraction twiddles W_k = exp(-2pii*k/nz) for k = 0..nz_c_val-1.
        let r2c_twiddles_64: Vec<Complex64> = {
            let mut v = Vec::with_capacity(nz_c_val);
            // SAFETY: every slot is overwritten in the loop before  is read.
            unsafe { v.set_len(nz_c_val) };
            for k in 0..nz_c_val {
                let a = -std::f64::consts::TAU * k as f64 / nz as f64;
                unsafe { *v.get_unchecked_mut(k) = Complex64::new(a.cos(), a.sin()) };
            }
            v
        };
        Self {
            nx,
            ny,
            nz,
            nz_c: nz_c_val,
            precision,
            twiddle_z_fwd_64: make64(nz, true),
            twiddle_z_inv_64: make64(nz, false),
            twiddle_y_fwd_64: make64(ny, true),
            twiddle_y_inv_64: make64(ny, false),
            twiddle_x_fwd_64: make64(nx, true),
            twiddle_x_inv_64: make64(nx, false),
            twiddle_z_fwd_32: make32(nz, true),
            twiddle_z_inv_32: make32(nz, false),
            twiddle_y_fwd_32: make32(ny, true),
            twiddle_y_inv_32: make32(ny, false),
            twiddle_x_fwd_32: make32(nx, true),
            twiddle_x_inv_32: make32(nx, false),
            scratch_y_64: std::sync::Mutex::new(uninit_copy_vec(vol)),
            scratch_x_64: std::sync::Mutex::new(uninit_copy_vec(vol)),
            scratch_y_32: std::sync::Mutex::new(uninit_copy_vec(vol)),
            scratch_x_32: std::sync::Mutex::new(uninit_copy_vec(vol)),
            twiddle_zh_fwd_64: make64(m, true),
            twiddle_zh_inv_64: make64(m, false),
            r2c_twiddles_64,
            scratch_r2c_y_64: std::sync::Mutex::new(uninit_copy_vec(r2c_vol)),
            scratch_r2c_x_64: std::sync::Mutex::new(uninit_copy_vec(r2c_vol)),
        }
    }

    /// Return the precision profile used by this plan.
    #[must_use]
    pub fn precision_profile(&self) -> PrecisionProfile {
        self.precision
    }

    /// Return the half-spectrum bookkeeping value `nz / 2 + 1`.
    ///
    /// No R2C half-spectrum transform is implemented: every forward and inverse
    /// transform always operates on the full `(nx, ny, nz)` complex array
    /// internally. This accessor does not describe an actual reduction in the Z
    /// dimension.
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

    /// Forward transform of a real 3D field.
    #[must_use]
    pub fn forward(&self, input: &Array3<f64>) -> Array3<Complex64> {
        self.forward_real_to_complex(input)
    }

    /// Inverse transform of a full complex 3D spectrum.
    #[must_use]
    pub fn inverse(&self, input: &Array3<Complex64>) -> Array3<f64> {
        self.inverse_complex_to_real(input)
    }

    /// Forward transform of a real field using generic storage dispatch.
    #[must_use]
    pub fn forward_typed<T: RealFftData>(&self, input: &Array3<T>) -> Array3<T::Spectrum> {
        T::forward_3d(self, input)
    }

    /// Inverse transform of a complex spectrum using generic storage dispatch.
    #[must_use]
    pub fn inverse_typed<T: RealFftData>(&self, input: &Array3<T::Spectrum>) -> Array3<T> {
        T::inverse_3d(self, input)
    }

    /// Forward transform of a real field into caller-owned typed spectrum storage.
    pub fn forward_typed_into<T: RealFftData>(
        &self,
        input: &Array3<T>,
        output: &mut Array3<T::Spectrum>,
    ) {
        T::forward_3d_into(self, input, output);
    }

    /// Inverse transform into caller-owned typed real storage and scratch spectrum.
    pub fn inverse_typed_into<T: RealFftData>(
        &self,
        input: &Array3<T::Spectrum>,
        output: &mut Array3<T>,
        scratch: &mut Array3<T::Spectrum>,
    ) {
        T::inverse_3d_into(self, input, output, scratch);
    }

    /// Forward transform of a complex field.
    #[must_use]
    pub fn forward_complex(&self, input: &Array3<Complex64>) -> Array3<Complex64> {
        let mut output = input.clone();
        self.forward_complex_inplace(&mut output);
        output
    }

    /// Inverse transform of a complex field.
    #[must_use]
    pub fn inverse_complex(&self, input: &Array3<Complex64>) -> Array3<Complex64> {
        let mut output = input.clone();
        self.inverse_complex_inplace(&mut output);
        output
    }

    /// Forward transform of a real field.
    #[must_use]
    pub fn forward_real_to_complex(&self, input: &Array3<f64>) -> Array3<Complex64> {
        let mut output = Array3::<Complex64>::from_shape_vec(
            (self.nx, self.ny, self.nz),
            uninit_copy_vec(self.nx * self.ny * self.nz),
        )
        .expect("uninit Complex64 3D buffer length must match plan shape");
        self.forward_real_to_complex_into_full(input, &mut output);
        output
    }

    /// Forward transform of a real field into a full complex spectrum buffer.
    pub fn forward_real_to_complex_into(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<Complex64>,
    ) {
        self.forward_real_to_complex_into_full(input, output);
    }

    /// Inverse transform of a full complex spectrum to a real field.
    #[must_use]
    pub fn inverse_complex_to_real(&self, input: &Array3<Complex64>) -> Array3<f64> {
        let mut output = Array3::<f64>::from_shape_vec(
            (self.nx, self.ny, self.nz),
            uninit_copy_vec(self.nx * self.ny * self.nz),
        )
        .expect("uninit f64 3D buffer length must match plan shape");
        self.inverse_complex_to_real_with_workspace(input, &mut output);
        output
    }

    /// Inverse transform into caller-owned output and scratch buffers.
    pub fn inverse_complex_to_real_into(
        &self,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        self.check_full_complex_shape(input.dim(), "inverse input");
        self.check_real_shape(output.dim(), "inverse output");
        self.check_full_complex_shape(scratch.dim(), "inverse scratch");
        scratch.assign(input);
        self.inverse_complex_inplace(scratch);
        Zip::from(output).and(scratch).for_each(|out, value| {
            *out = value.re;
        });
    }

    /// Forward transform of a complex field in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array3<Complex64>) {
        self.check_full_complex_shape(data.dim(), "forward input");
        self.forward_complex_axis_pass(data);
    }

    /// Inverse transform of a complex field in-place with FFTW-compatible normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<Complex64>) {
        self.check_full_complex_shape(data.dim(), "inverse input");
        self.inverse_complex_axis_pass(data);
    }

    /// Forward transform of a real field stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array3<f32>) -> Array3<Complex32> {
        self.forward_real32(input)
    }

    /// Inverse transform of an `f32`-storage complex spectrum.
    #[must_use]
    pub(crate) fn inverse_f32(&self, input: &Array3<Complex32>) -> Array3<f32> {
        self.inverse_real32(input)
    }

    /// Forward transform of a real `f32` field into caller-owned spectrum storage.
    pub(crate) fn forward_f32_into(&self, input: &Array3<f32>, output: &mut Array3<Complex32>) {
        self.forward_real32_into(input, output);
    }

    /// Inverse transform of an `f32` spectrum into caller-owned real storage.
    pub(crate) fn inverse_f32_into(
        &self,
        input: &Array3<Complex32>,
        output: &mut Array3<f32>,
        scratch: &mut Array3<Complex32>,
    ) {
        self.inverse_real32_into(input, output, scratch);
    }

    /// Forward transform of a real field stored as `f16`.
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array3<f16>) -> Array3<Complex32> {
        self.forward_real32(input)
    }

    /// Inverse transform of a complex spectrum to `f16` storage.
    #[must_use]
    pub(crate) fn inverse_f16(&self, input: &Array3<Complex32>) -> Array3<f16> {
        self.inverse_real32(input)
    }

    /// Forward transform of a real `f16` field into caller-owned `f32` spectrum storage.
    pub(crate) fn forward_f16_into(&self, input: &Array3<f16>, output: &mut Array3<Complex32>) {
        self.forward_real32_into(input, output);
    }

    /// Inverse transform of a `Complex32` spectrum into caller-owned `f16` storage.
    pub(crate) fn inverse_f16_into(
        &self,
        input: &Array3<Complex32>,
        output: &mut Array3<f16>,
        scratch: &mut Array3<Complex32>,
    ) {
        self.inverse_real32_into(input, output, scratch);
    }

    // forward_r2c, forward_r2c_into, inverse_c2r, inverse_c2r_into, and
    // inverse_c2r_into_with_scratch are implemented in r2c.rs.

    pub(super) fn check_real_shape(&self, dim: (usize, usize, usize), label: &str) {
        assert_eq!(dim, (self.nx, self.ny, self.nz), "{label} shape mismatch");
    }

    pub(super) fn check_full_complex_shape(&self, dim: (usize, usize, usize), label: &str) {
        assert_eq!(dim, (self.nx, self.ny, self.nz), "{label} shape mismatch");
    }

    pub(super) fn check_half_complex_shape(&self, dim: (usize, usize, usize), label: &str) {
        assert_eq!(
            dim,
            (self.nx, self.ny, self.nz_c),
            "{label} half-spectrum shape mismatch"
        );
    }

    pub(super) fn inverse_complex_to_real_with_workspace(
        &self,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
    ) {
        self.check_full_complex_shape(input.dim(), "inverse input");
        self.check_real_shape(output.dim(), "inverse output");
        let transformed = self.inverse_complex(input);
        Zip::from(output).and(&transformed).for_each(|out, value| {
            *out = value.re;
        });
    }
}

mod axis_pass_f32;
mod axis_pass_f64;
mod precision_bridge;
mod r2c;
mod r2c_axis_pass;
mod r2c_row;
#[cfg(test)]
mod tests;
