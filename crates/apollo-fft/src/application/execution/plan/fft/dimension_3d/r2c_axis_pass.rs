use super::FftPlan3D;
use super::{GATHER_TILE, RAYON_THRESHOLD};
use crate::application::execution::kernel::mixed_radix::{
    forward_inplace_64_with_twiddles, inverse_inplace_64_with_twiddles,
};
use crate::application::execution::kernel::{fft_forward, fft_inverse};
use num_complex::Complex64;
use rayon::prelude::*;

impl FftPlan3D {
    /// Y-axis (axis-1) complex FFT/IFFT pass on `(nx, ny, nz_c)` half-spectrum data.
    ///
    /// Operates on a flat slice of length `nx * ny * nz_c` laid out in C order
    /// `[i][j][k]`, applying length-ny transforms along axis 1.
    pub(super) fn r2c_axis1_pass_64(&self, data: &mut [Complex64], forward: bool) {
        if self.ny <= 1 {
            return;
        }
        let nz_c = self.nz_c;
        let nx = self.nx;
        let ny = self.ny;
        let total = nx * ny * nz_c;
        let mut scratch = self
            .scratch_r2c_y_64
            .lock()
            .expect("scratch_r2c_y_64 mutex poisoned");

        // Cache-blocked gather: data[i,j,k] -> scratch[i,k,j] (transpose j<->k).
        for i in 0..nx {
            for j_t in (0..ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(ny);
                for k_t in (0..nz_c).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(nz_c);
                    for j in j_t..j_end {
                        let src = (i * ny + j) * nz_c;
                        for k in k_t..k_end {
                            scratch[(i * nz_c + k) * ny + j] = data[src + k];
                        }
                    }
                }
            }
        }

        let lane_fn = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_y_fwd_64,
            &self.twiddle_y_inv_64,
        ) {
            (true, Some(tw), _) => forward_inplace_64_with_twiddles(lane, Some(tw.as_ref())),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, Some(tw.as_ref())),
            _ => {
                if forward {
                    fft_forward(lane);
                } else {
                    fft_inverse(lane);
                }
            }
        };

        if total > RAYON_THRESHOLD {
            scratch[..total].par_chunks_mut(ny).for_each(lane_fn);
        } else {
            scratch[..total].chunks_mut(ny).for_each(lane_fn);
        }

        // Cache-blocked scatter: scratch[i,k,j] -> data[i,j,k].
        for i in 0..nx {
            for j_t in (0..ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(ny);
                for k_t in (0..nz_c).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(nz_c);
                    for j in j_t..j_end {
                        let dst = (i * ny + j) * nz_c;
                        for k in k_t..k_end {
                            data[dst + k] = scratch[(i * nz_c + k) * ny + j];
                        }
                    }
                }
            }
        }
    }

    /// X-axis (axis-0) complex FFT/IFFT pass on `(nx, ny, nz_c)` half-spectrum data.
    pub(super) fn r2c_axis0_pass_64(&self, data: &mut [Complex64], forward: bool) {
        if self.nx <= 1 {
            return;
        }
        let nz_c = self.nz_c;
        let nx = self.nx;
        let ny = self.ny;
        let total = nx * ny * nz_c;
        let mut scratch = self
            .scratch_r2c_x_64
            .lock()
            .expect("scratch_r2c_x_64 mutex poisoned");

        // Cache-blocked gather: data[i,j,k] -> scratch[j,k,i].
        for i in 0..nx {
            let src_base = i * ny * nz_c;
            for j_t in (0..ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(ny);
                for k_t in (0..nz_c).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(nz_c);
                    for j in j_t..j_end {
                        let src = src_base + j * nz_c;
                        for k in k_t..k_end {
                            scratch[(j * nz_c + k) * nx + i] = data[src + k];
                        }
                    }
                }
            }
        }

        let lane_fn = |lane: &mut [Complex64]| match (
            forward,
            &self.twiddle_x_fwd_64,
            &self.twiddle_x_inv_64,
        ) {
            (true, Some(tw), _) => forward_inplace_64_with_twiddles(lane, Some(tw.as_ref())),
            (false, _, Some(tw)) => inverse_inplace_64_with_twiddles(lane, Some(tw.as_ref())),
            _ => {
                if forward {
                    fft_forward(lane);
                } else {
                    fft_inverse(lane);
                }
            }
        };

        if total > RAYON_THRESHOLD {
            scratch[..total].par_chunks_mut(nx).for_each(lane_fn);
        } else {
            scratch[..total].chunks_mut(nx).for_each(lane_fn);
        }

        // Cache-blocked scatter: scratch[j,k,i] -> data[i,j,k].
        for i in 0..nx {
            let dst_base = i * ny * nz_c;
            for j_t in (0..ny).step_by(GATHER_TILE) {
                let j_end = (j_t + GATHER_TILE).min(ny);
                for k_t in (0..nz_c).step_by(GATHER_TILE) {
                    let k_end = (k_t + GATHER_TILE).min(nz_c);
                    for j in j_t..j_end {
                        let dst = dst_base + j * nz_c;
                        for k in k_t..k_end {
                            data[dst + k] = scratch[(j * nz_c + k) * nx + i];
                        }
                    }
                }
            }
        }
    }
}
