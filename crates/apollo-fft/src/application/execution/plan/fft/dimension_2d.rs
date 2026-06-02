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

use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use crate::application::execution::plan::fft::workspace::{
    uninit_copy_vec, UninitWorkspaceElement,
};
use crate::domain::metadata::shape::Shape2D;
use ndarray::{Array2, Axis};
use num_complex::Complex;
use std::sync::Arc;

/// Use rayon parallel iteration when total elements exceed this threshold.
/// Below the threshold, sequential iteration avoids rayon task-spawn overhead
/// that dominates for small matrices (e.g. 32×32 = 1024 elements).
const RAYON_THRESHOLD: usize = 32768;

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
    scratch_col: std::sync::Mutex<Vec<F::Complex>>,
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
    Complex<F>: UninitWorkspaceElement,
{
    /// Create a new 2D plan.
    #[must_use]
    pub fn new(shape: Shape2D) -> Self {
        let (nx, ny) = (shape.nx, shape.ny);
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
        Self {
            nx,
            ny,
            twiddle_row_fwd: make(ny, true),
            twiddle_row_inv: make(ny, false),
            twiddle_col_fwd: make(nx, true),
            twiddle_col_inv: make(nx, false),
            scratch_col: std::sync::Mutex::new(uninit_copy_vec(nx * ny)),
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
            data.dim(),
            (self.nx, self.ny),
            "complex forward shape mismatch"
        );
        self.axis_pass_complex(data, Axis(1), true);
        self.axis_pass_complex(data, Axis(0), true);
    }

    /// Inverse transform of a complex array in-place with normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array2<F::Complex>) {
        assert_eq!(
            data.dim(),
            (self.nx, self.ny),
            "complex inverse shape mismatch"
        );
        self.axis_pass_complex(data, Axis(0), false);
        self.axis_pass_complex(data, Axis(1), false);
    }

    fn axis_pass_complex(&self, data: &mut Array2<F::Complex>, axis: Axis, forward: bool) {
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

    fn axis1_pass_complex(&self, data: &mut Array2<F::Complex>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D complex data must be contiguous");
        let lane_fn =
            |lane: &mut [F::Complex]| match (forward, &self.twiddle_row_fwd, &self.twiddle_row_inv)
            {
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
            moirai::for_each_chunk_mut_with::<moirai::Adaptive, _, _>(
                data_slice, self.ny, lane_fn,
            );
        } else {
            data_slice.chunks_mut(self.ny).for_each(lane_fn);
        }
    }

    fn axis0_pass_complex(&self, data: &mut Array2<F::Complex>, forward: bool) {
        let data_slice = data
            .as_slice_memory_order_mut()
            .expect("2D complex data must be contiguous");
        let mut scratch = self.scratch_col.lock().expect("scratch_col mutex poisoned");
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
        let lane_fn =
            |lane: &mut [F::Complex]| match (forward, &self.twiddle_col_fwd, &self.twiddle_col_inv)
            {
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
            moirai::for_each_chunk_mut_with::<moirai::Adaptive, _, _>(
                &mut scratch[..], self.nx, lane_fn,
            );
        } else {
            scratch.chunks_mut(self.nx).for_each(lane_fn);
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
}
