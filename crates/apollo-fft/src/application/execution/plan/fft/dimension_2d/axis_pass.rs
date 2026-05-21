//! Axis-pass implementations for `FftPlan2D`.
//!
//! Contains the internal `axis1_pass_complex`, `axis0_pass_complex`,
//! `axis1_pass_complex_f32`, and `axis0_pass_complex_f32` methods, plus the
//! two inplace dispatch helpers. All methods are on `super::FftPlan2D`.

use super::{FftPlan2D, RAYON_THRESHOLD, TRANSPOSE_TILE};
use crate::application::execution::kernel::mixed_radix::{
    forward_inplace_32_with_twiddles, forward_inplace_64_with_twiddles,
    inverse_inplace_32_with_twiddles, inverse_inplace_64_with_twiddles,
};
use crate::application::execution::kernel::{fft_forward, fft_inverse};
use ndarray::{Array2, Axis};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

impl FftPlan2D {
    pub(super) fn axis_pass_complex(
        &self,
        data: &mut Array2<Complex64>,
        axis: Axis,
        forward: bool,
    ) {
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

    pub(super) fn axis1_pass_complex(&self, data: &mut Array2<Complex64>, forward: bool) {
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

    pub(super) fn axis0_pass_complex(&self, data: &mut Array2<Complex64>, forward: bool) {
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

    pub(super) fn forward_complex_inplace_f32(&self, data: &mut Array2<Complex32>) {
        self.axis_pass_complex_f32(data, Axis(1), true);
        self.axis_pass_complex_f32(data, Axis(0), true);
    }

    pub(super) fn inverse_complex_inplace_f32(&self, data: &mut Array2<Complex32>) {
        self.axis_pass_complex_f32(data, Axis(0), false);
        self.axis_pass_complex_f32(data, Axis(1), false);
    }

    pub(super) fn axis_pass_complex_f32(
        &self,
        data: &mut Array2<Complex32>,
        axis: Axis,
        forward: bool,
    ) {
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

    pub(super) fn axis1_pass_complex_f32(&self, data: &mut Array2<Complex32>, forward: bool) {
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

    pub(super) fn axis0_pass_complex_f32(&self, data: &mut Array2<Complex32>, forward: bool) {
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
