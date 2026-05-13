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
//! Bluestein plan paths. Contiguous innermost-axis passes mutate row chunks in
//! place, while non-contiguous passes gather lanes into scratch buffers before
//! scattering them back.

use crate::application::execution::kernel::mixed_radix::{
    cached_twiddle_fwd_32, cached_twiddle_fwd_64, cached_twiddle_inv_32, cached_twiddle_inv_64,
    forward_inplace_32_with_twiddles, forward_inplace_64_with_twiddles,
    inverse_inplace_32_with_twiddles, inverse_inplace_64_with_twiddles,
};
use crate::application::execution::kernel::{fft_forward, fft_inverse};
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::application::execution::plan::fft::workspace::{
    uninit_copy_vec, UninitWorkspaceElement,
};

/// Use rayon parallel iteration when total elements exceed this threshold.
/// Below the threshold, sequential iteration avoids rayon task-spawn overhead
/// that dominates for small matrices (e.g. 32×32 = 1024 elements).
const RAYON_THRESHOLD: usize = 32768; // 32K Complex64 ≈ 512 KB

/// Tile size for cache-blocked transpose.
/// A 32×32 tile of Complex64 = 8 KB, fitting comfortably in L1 (32–48 KB).
const TRANSPOSE_TILE: usize = 32;
use crate::domain::metadata::precision::PrecisionProfile;
use crate::domain::metadata::shape::Shape2D;
use half::f16;
use ndarray::{Array2, Axis, Zip};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
use std::sync::Arc;

trait Plan2dReal32: Copy + UninitWorkspaceElement {
    const NATIVE_PROFILE: PrecisionProfile;

    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
    fn from_f32(value: f32) -> Self;
    fn from_f64(value: f64) -> Self;
}

impl Plan2dReal32 for f32 {
    const NATIVE_PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl Plan2dReal32 for f16 {
    const NATIVE_PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        Self::from_f32(value as f32)
    }
}

/// Reusable 2D FFT plan.
///
/// # Precomputed twiddle tables and scratch buffers
///
/// For power-of-two axis lengths, the plan precomputes contiguous per-stage
/// twiddle tables at construction time (total N-1 entries per axis, per
/// direction). All row and column butterfly passes use these tables via
/// `forward_inplace_64_with_twiddles` / `inverse_inplace_64_with_twiddles`,
/// reading twiddles sequentially with no stride. This eliminates:
/// 1. The per-lane twiddle-table `Vec` allocation that the naïve kernel
///    performs on every generic `fft_forward` call (O(ny) alloc × nx rows).
/// 2. The strided access pattern (`T[j * stride]`) that causes L1 cache
///    misses for large N.
///
/// For non-power-of-two axis lengths the plan falls back to generic
/// `fft_forward` / `fft_inverse`, which use Bluestein with per-call scratch
/// allocation.
///
/// The column pass must transpose, FFT, then scatter; this requires an
/// auxiliary buffer of `nx * ny` entries. This scratch buffer is preallocated
/// at plan construction time and reused for every column pass via a Mutex,
/// eliminating the per-call `Vec<Complex64>` allocation that previously
/// occurred on every axis-0 pass.
pub struct FftPlan2D {
    nx: usize,
    ny: usize,
    precision: PrecisionProfile,
    /// Per-stage forward twiddle table for row (axis-1) passes of length `ny`.
    /// `Some` iff `ny` is a power of two and `ny > 1`.
    twiddle_row_fwd_64: Option<Arc<[Complex64]>>,
    /// Per-stage inverse twiddle table for row (axis-1) passes of length `ny`.
    twiddle_row_inv_64: Option<Arc<[Complex64]>>,
    /// Per-stage forward twiddle table for column (axis-0) passes of length `nx`.
    twiddle_col_fwd_64: Option<Arc<[Complex64]>>,
    /// Per-stage inverse twiddle table for column (axis-0) passes of length `nx`.
    twiddle_col_inv_64: Option<Arc<[Complex64]>>,
    /// f32 variants of the above four.
    twiddle_row_fwd_32: Option<Arc<[Complex32]>>,
    twiddle_row_inv_32: Option<Arc<[Complex32]>>,
    twiddle_col_fwd_32: Option<Arc<[Complex32]>>,
    twiddle_col_inv_32: Option<Arc<[Complex32]>>,
    /// Preallocated scratch buffer for the column transpose-FFT-scatter pass.
    /// Holds `nx * ny` Complex64 entries; protected by Mutex for shared plan reuse.
    scratch_col_64: std::sync::Mutex<Vec<Complex64>>,
    /// Preallocated scratch buffer for the f32 column pass.
    scratch_col_32: std::sync::Mutex<Vec<Complex32>>,
}

impl std::fmt::Debug for FftPlan2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan2D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("precision", &self.precision)
            .finish()
    }
}

impl FftPlan2D {
    /// Create a new 2D plan.
    #[must_use]
    pub fn new(shape: Shape2D) -> Self {
        Self::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 2D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(shape: Shape2D, precision: PrecisionProfile) -> Self {
        let (nx, ny) = (shape.nx, shape.ny);
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
        Self {
            nx,
            ny,
            precision,
            twiddle_row_fwd_64: make64(ny, true),
            twiddle_row_inv_64: make64(ny, false),
            twiddle_col_fwd_64: make64(nx, true),
            twiddle_col_inv_64: make64(nx, false),
            twiddle_row_fwd_32: make32(ny, true),
            twiddle_row_inv_32: make32(ny, false),
            twiddle_col_fwd_32: make32(nx, true),
            twiddle_col_inv_32: make32(nx, false),
            scratch_col_64: std::sync::Mutex::new(uninit_copy_vec(nx * ny)),
            scratch_col_32: std::sync::Mutex::new(uninit_copy_vec(nx * ny)),
        }
    }

    /// Return the precision profile used by this plan.
    #[must_use]
    pub fn precision_profile(&self) -> PrecisionProfile {
        self.precision
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape2D {
        Shape2D {
            nx: self.nx,
            ny: self.ny,
        }
    }

    /// Forward transform of a real array.
    #[must_use]
    pub fn forward(&self, input: &Array2<f64>) -> Array2<Complex64> {
        self.forward_real_to_complex(input)
    }

    /// Inverse transform returning a real array.
    #[must_use]
    pub fn inverse(&self, input: &Array2<Complex64>) -> Array2<f64> {
        self.inverse_complex_to_real(input)
    }

    /// Forward transform of a real array using generic storage dispatch.
    #[must_use]
    pub fn forward_typed<T: RealFftData>(&self, input: &Array2<T>) -> Array2<T::Spectrum> {
        T::forward_2d(self, input)
    }

    /// Inverse transform of a complex spectrum using generic storage dispatch.
    #[must_use]
    pub fn inverse_typed<T: RealFftData>(&self, input: &Array2<T::Spectrum>) -> Array2<T> {
        T::inverse_2d(self, input)
    }

    /// Forward transform of a real array.
    #[must_use]
    pub fn forward_real_to_complex(&self, input: &Array2<f64>) -> Array2<Complex64> {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "forward input shape mismatch"
        );
        let mut data = input.mapv(|value| Complex64::new(value, 0.0));
        self.forward_complex_inplace(&mut data);
        data
    }

    /// Forward transform of a real array into a caller-supplied complex buffer.
    pub fn forward_real_to_complex_into(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<Complex64>,
    ) {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "forward input shape mismatch"
        );
        assert_eq!(
            output.dim(),
            (self.nx, self.ny),
            "forward output shape mismatch"
        );
        Zip::from(&mut *output).and(input).for_each(|out, &value| {
            *out = Complex64::new(value, 0.0);
        });
        self.forward_complex_inplace(output);
    }

    /// Inverse transform returning a real array.
    #[must_use]
    pub fn inverse_complex_to_real(&self, input: &Array2<Complex64>) -> Array2<f64> {
        let mut output = Array2::<f64>::zeros((self.nx, self.ny));
        self.inverse_complex_to_real_with_workspace(input, &mut output);
        output
    }

    /// Inverse transform into caller-owned real and scratch buffers.
    pub fn inverse_complex_to_real_into(
        &self,
        input: &Array2<Complex64>,
        output: &mut Array2<f64>,
        scratch: &mut Array2<Complex64>,
    ) {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "inverse input shape mismatch"
        );
        assert_eq!(
            output.dim(),
            (self.nx, self.ny),
            "inverse output shape mismatch"
        );
        assert_eq!(
            scratch.dim(),
            (self.nx, self.ny),
            "inverse scratch shape mismatch"
        );
        scratch.assign(input);
        self.inverse_complex_inplace(scratch);
        Zip::from(output).and(scratch).for_each(|out, value| {
            *out = value.re;
        });
    }

    fn inverse_complex_to_real_with_workspace(
        &self,
        input: &Array2<Complex64>,
        output: &mut Array2<f64>,
    ) {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "inverse input shape mismatch"
        );
        assert_eq!(
            output.dim(),
            (self.nx, self.ny),
            "inverse output shape mismatch"
        );
        let mut transformed = input.clone();
        self.inverse_complex_inplace(&mut transformed);
        Zip::from(output).and(&transformed).for_each(|out, value| {
            *out = value.re;
        });
    }

    /// Forward transform of a complex array in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array2<Complex64>) {
        assert_eq!(
            data.dim(),
            (self.nx, self.ny),
            "complex forward shape mismatch"
        );
        self.axis_pass_complex(data, Axis(1), true);
        self.axis_pass_complex(data, Axis(0), true);
    }

    /// Inverse transform of a complex array in-place with normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array2<Complex64>) {
        assert_eq!(
            data.dim(),
            (self.nx, self.ny),
            "complex inverse shape mismatch"
        );
        self.axis_pass_complex(data, Axis(0), false);
        self.axis_pass_complex(data, Axis(1), false);
    }

    /// Forward transform of a real array stored as `f32`.
    #[must_use]
    pub(crate) fn forward_f32(&self, input: &Array2<f32>) -> Array2<Complex32> {
        self.forward_real32(input)
    }

    /// Inverse transform of an `f32`-storage complex spectrum.
    #[must_use]
    pub(crate) fn inverse_f32(&self, input: &Array2<Complex32>) -> Array2<f32> {
        self.inverse_real32(input)
    }

    /// Forward transform of a real array stored as `f16`.
    #[must_use]
    pub(crate) fn forward_f16(&self, input: &Array2<f16>) -> Array2<Complex32> {
        self.forward_real32(input)
    }

    /// Inverse transform of a complex spectrum to `f16` storage.
    #[must_use]
    pub(crate) fn inverse_f16(&self, input: &Array2<Complex32>) -> Array2<f16> {
        self.inverse_real32(input)
    }

    fn forward_real32<T: Plan2dReal32>(&self, input: &Array2<T>) -> Array2<Complex32> {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "forward input shape mismatch"
        );
        if self.precision == T::NATIVE_PROFILE {
            let mut data = self.pack_real32_complex(input);
            self.forward_complex_inplace_f32(&mut data);
            data
        } else {
            let promoted = input.mapv(T::to_f64);
            self.forward_real_to_complex(&promoted)
                .mapv(|value| Complex32::new(value.re as f32, value.im as f32))
        }
    }

    fn inverse_real32<T: Plan2dReal32>(&self, input: &Array2<Complex32>) -> Array2<T> {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny),
            "inverse input shape mismatch"
        );
        if self.precision == T::NATIVE_PROFILE {
            let mut data = input.clone();
            self.inverse_complex_inplace_f32(&mut data);
            self.project_real32(data)
        } else {
            let promoted =
                input.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
            self.inverse_complex_to_real(&promoted).mapv(T::from_f64)
        }
    }

    fn pack_real32_complex<T: Plan2dReal32>(&self, input: &Array2<T>) -> Array2<Complex32> {
        let mut values = uninit_copy_vec(input.len());
        for (slot, value) in values.iter_mut().zip(input.iter()) {
            *slot = Complex32::new(value.to_f32(), 0.0);
        }
        Array2::from_shape_vec((self.nx, self.ny), values)
            .expect("packed 2D real32 buffer length must match plan shape")
    }

    fn project_real32<T: Plan2dReal32>(&self, input: Array2<Complex32>) -> Array2<T> {
        let mut values = uninit_copy_vec(input.len());
        for (slot, value) in values.iter_mut().zip(input.iter()) {
            *slot = T::from_f32(value.re);
        }
        Array2::from_shape_vec((self.nx, self.ny), values)
            .expect("projected 2D real32 buffer length must match plan shape")
    }

    fn axis_pass_complex(&self, data: &mut Array2<Complex64>, axis: Axis, forward: bool) {
        if axis.index() == 1 {
            self.axis1_pass_complex(data, forward);
            return;
        }
        if axis.index() == 0 {
            self.axis0_pass_complex(data, forward);
            return;
        }

        unreachable!("2D FFT axis index must be 0 or 1");
    }

    fn axis1_pass_complex(&self, data: &mut Array2<Complex64>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D complex data must be contiguous");
        let lane_fn_64 = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_row_fwd_64,
            &self.twiddle_row_inv_64,
        ) {
            (true, Some(tw), _) => forward_inplace_64_with_twiddles(lane, Some(tw.as_ref())),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, Some(tw.as_ref())),
            _ => {
                if forward {
                    fft_forward(lane)
                } else {
                    fft_inverse(lane)
                }
            }
        };
        if data_slice.len() > RAYON_THRESHOLD {
            data_slice.par_chunks_mut(self.ny).for_each(lane_fn_64);
        } else {
            data_slice.chunks_mut(self.ny).for_each(lane_fn_64);
        }
    }

    fn axis0_pass_complex(&self, data: &mut Array2<Complex64>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D complex data must be contiguous");
        let mut scratch = self
            .scratch_col_64
            .lock()
            .expect("scratch_col_64 mutex poisoned");
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
        let lane_fn_64 = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_col_fwd_64,
            &self.twiddle_col_inv_64,
        ) {
            (true, Some(tw), _) => forward_inplace_64_with_twiddles(lane, Some(tw.as_ref())),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, Some(tw.as_ref())),
            _ => {
                if forward {
                    fft_forward(lane)
                } else {
                    fft_inverse(lane)
                }
            }
        };
        if scratch.len() > RAYON_THRESHOLD {
            scratch.par_chunks_mut(self.nx).for_each(lane_fn_64);
        } else {
            scratch.chunks_mut(self.nx).for_each(lane_fn_64);
        }
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
    }

    fn forward_complex_inplace_f32(&self, data: &mut Array2<Complex32>) {
        self.axis_pass_complex_f32(data, Axis(1), true);
        self.axis_pass_complex_f32(data, Axis(0), true);
    }

    fn inverse_complex_inplace_f32(&self, data: &mut Array2<Complex32>) {
        self.axis_pass_complex_f32(data, Axis(0), false);
        self.axis_pass_complex_f32(data, Axis(1), false);
    }

    fn axis_pass_complex_f32(&self, data: &mut Array2<Complex32>, axis: Axis, forward: bool) {
        if axis.index() == 1 {
            self.axis1_pass_complex_f32(data, forward);
            return;
        }
        if axis.index() == 0 {
            self.axis0_pass_complex_f32(data, forward);
            return;
        }

        unreachable!("2D FFT axis index must be 0 or 1");
    }

    fn axis1_pass_complex_f32(&self, data: &mut Array2<Complex32>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D f32 complex data must be contiguous");
        let lane_fn_32 = |lane: &mut [Complex32]| match (
            forward,
            &self.twiddle_row_fwd_32,
            &self.twiddle_row_inv_32,
        ) {
            (true, Some(tw), _) => forward_inplace_32_with_twiddles(lane, Some(tw.as_ref())),
            (false, _, Some(tw)) => inverse_inplace_32_with_twiddles(lane, Some(tw.as_ref())),
            _ => {
                if forward {
                    fft_forward(lane)
                } else {
                    fft_inverse(lane)
                }
            }
        };
        if data_slice.len() > RAYON_THRESHOLD {
            data_slice.par_chunks_mut(self.ny).for_each(lane_fn_32);
        } else {
            data_slice.chunks_mut(self.ny).for_each(lane_fn_32);
        }
    }

    fn axis0_pass_complex_f32(&self, data: &mut Array2<Complex32>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D f32 complex data must be contiguous");
        let mut scratch = self
            .scratch_col_32
            .lock()
            .expect("scratch_col_32 mutex poisoned");
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
        let lane_fn_32 = |lane: &mut [Complex32]| match (
            forward,
            &self.twiddle_col_fwd_32,
            &self.twiddle_col_inv_32,
        ) {
            (true, Some(tw), _) => forward_inplace_32_with_twiddles(lane, Some(tw.as_ref())),
            (false, _, Some(tw)) => inverse_inplace_32_with_twiddles(lane, Some(tw.as_ref())),
            _ => {
                if forward {
                    fft_forward(lane)
                } else {
                    fft_inverse(lane)
                }
            }
        };
        if scratch.len() > RAYON_THRESHOLD {
            scratch.par_chunks_mut(self.nx).for_each(lane_fn_32);
        } else {
            scratch.chunks_mut(self.nx).for_each(lane_fn_32);
        }
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn roundtrip_recovers_asymmetric_inputs() {
        for &(nx, ny) in &[(7, 13), (13, 7), (16, 24), (24, 16), (120, 360)] {
            let shape = Shape2D::new(nx, ny).expect("test dimensions are non-zero");
            let plan = FftPlan2D::new(shape);
            let input = Array2::from_shape_fn((nx, ny), |(i, j)| {
                let x = i as f64 / nx as f64;
                let y = j as f64 / ny as f64;
                (std::f64::consts::TAU * x).sin()
                    + 0.5 * (2.0 * std::f64::consts::TAU * y).cos()
                    + 0.125 * ((i * 3 + j * 5) as f64).sin()
            });

            let spectrum = plan.forward(&input);
            let recovered = plan.inverse(&spectrum);

            for ((i, j), expected) in input.indexed_iter() {
                assert_abs_diff_eq!(recovered[(i, j)], *expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn complex_inverse_inplace_is_normalized() {
        let nx = 5;
        let ny = 9;
        let shape = Shape2D::new(nx, ny).expect("test dimensions are non-zero");
        let plan = FftPlan2D::new(shape);
        let input = Array2::from_shape_fn((nx, ny), |(i, j)| {
            Complex64::new((i + 2 * j) as f64, (2 * i + j) as f64 / 3.0)
        });

        let mut spectrum = input.clone();
        plan.forward_complex_inplace(&mut spectrum);
        plan.inverse_complex_inplace(&mut spectrum);

        for ((i, j), expected) in input.indexed_iter() {
            assert_abs_diff_eq!(spectrum[(i, j)].re, expected.re, epsilon = 1e-9);
            assert_abs_diff_eq!(spectrum[(i, j)].im, expected.im, epsilon = 1e-9);
        }
    }
}
