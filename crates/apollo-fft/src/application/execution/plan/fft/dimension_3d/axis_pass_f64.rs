use super::FftPlan3D;
use super::{GATHER_TILE, RAYON_THRESHOLD};
use crate::application::execution::kernel::mixed_radix::{
    forward_inplace_64_with_twiddles, inverse_inplace_64_with_twiddles,
};
use crate::application::execution::kernel::{fft_forward, fft_inverse};
use ndarray::{Array3, Axis};
use num_complex::Complex64;
use rayon::prelude::*;

impl FftPlan3D {
    pub(super) fn axis_pass_complex(
        &self,
        data: &mut Array3<Complex64>,
        axis: Axis,
        forward: bool,
    ) {
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

    fn axis1_pass_complex(&self, data: &mut Array3<Complex64>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        let mut scratch = self
            .scratch_y_64
            .lock()
            .expect("scratch_y_64 mutex poisoned");
        // Cache-blocked gather: data[i,j,k] (row-major) -> scratch[i,k,j].
        // j-outer / k-inner: reads data_slice[i][j][k] sequentially in k (stride 1),
        // writes scratch[i][k][j] with stride ny. Strided stores buffer in the store
        // queue; strided loads stall the pipeline.
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
        let lane_fn_64 = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_y_fwd_64,
            &self.twiddle_y_inv_64,
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
            scratch.par_chunks_mut(self.ny).for_each(lane_fn_64);
        } else {
            scratch.chunks_mut(self.ny).for_each(lane_fn_64);
        }
        // Cache-blocked scatter: scratch[i,k,j] -> data[i,j,k].
        // j-outer / k-inner: writes data_slice[i][j][k] sequentially in k (stride 1),
        // reads scratch[i][k][j] with stride ny.
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

    fn axis0_pass_complex(&self, data: &mut Array3<Complex64>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        let mut scratch = self
            .scratch_x_64
            .lock()
            .expect("scratch_x_64 mutex poisoned");
        // Cache-blocked gather: data[i,j,k] -> scratch[j,k,i].
        // i-outer loop: for each i reads data_slice[i][j][k] sequentially in k (stride 1),
        // writes scratch[j][k][i] with stride nx. Strided stores buffer in the store
        // queue; strided loads (stride ny*nz) stall the pipeline catastrophically.
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
        let lane_fn_64 = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_x_fwd_64,
            &self.twiddle_x_inv_64,
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
        // Cache-blocked scatter: scratch[j,k,i] -> data[i,j,k].
        // i-outer loop: writes data_slice[i][j][k] sequentially in k (stride 1),
        // reads scratch[j][k][i] with stride nx.
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

    fn axis2_pass_complex(&self, data: &mut Array3<Complex64>, forward: bool) {
        if self.nz <= 1 {
            return;
        }
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("3D complex data must be contiguous");
        let lane_fn_64 = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_z_fwd_64,
            &self.twiddle_z_inv_64,
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
            data_slice.par_chunks_mut(self.nz).for_each(lane_fn_64);
        } else {
            data_slice.chunks_mut(self.nz).for_each(lane_fn_64);
        }
    }
}
