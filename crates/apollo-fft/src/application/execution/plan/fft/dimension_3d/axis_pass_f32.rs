use super::FftPlan3D;
use super::{GATHER_TILE, RAYON_THRESHOLD};
use crate::application::execution::kernel::mixed_radix::{
    forward_inplace_32_with_twiddles, inverse_inplace_32_with_twiddles,
};
use crate::application::execution::kernel::{fft_forward, fft_inverse};
use ndarray::{Array3, Axis};
use num_complex::Complex32;
use rayon::prelude::*;

impl FftPlan3D {
    pub(super) fn axis_pass_complex_f32(
        &self,
        data: &mut Array3<Complex32>,
        axis: Axis,
        forward: bool,
    ) {
        if data.len_of(axis) <= 1 {
            return;
        }
        if axis.index() == 2 {
            self.axis2_pass_complex_f32(data, forward);
            return;
        }
        if axis.index() == 1 {
            self.axis1_pass_complex_f32(data, forward);
            return;
        }
        if axis.index() == 0 {
            self.axis0_pass_complex_f32(data, forward);
        }
    }

    fn axis1_pass_complex_f32(&self, data: &mut Array3<Complex32>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D f32 complex data must be contiguous");
        let mut scratch = self
            .scratch_y_32
            .lock()
            .expect("scratch_y_32 mutex poisoned");
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
        let lane_fn_32 = |lane: &mut [Complex32]| match (
            forward,
            &self.twiddle_y_fwd_32,
            &self.twiddle_y_inv_32,
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
            scratch.par_chunks_mut(self.ny).for_each(lane_fn_32);
        } else {
            scratch.chunks_mut(self.ny).for_each(lane_fn_32);
        }
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

    fn axis0_pass_complex_f32(&self, data: &mut Array3<Complex32>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D f32 complex data must be contiguous");
        let mut scratch = self
            .scratch_x_32
            .lock()
            .expect("scratch_x_32 mutex poisoned");
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
        let lane_fn_32 = |lane: &mut [Complex32]| match (
            forward,
            &self.twiddle_x_fwd_32,
            &self.twiddle_x_inv_32,
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

    fn axis2_pass_complex_f32(&self, data: &mut Array3<Complex32>, forward: bool) {
        if self.nz <= 1 {
            return;
        }
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D f32 complex data must be contiguous");
        let lane_fn_32 = |lane: &mut [Complex32]| match (
            forward,
            &self.twiddle_z_fwd_32,
            &self.twiddle_z_inv_32,
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
            data_slice.par_chunks_mut(self.nz).for_each(lane_fn_32);
        } else {
            data_slice.chunks_mut(self.nz).for_each(lane_fn_32);
        }
    }
}
