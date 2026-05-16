//! Cache-optimal Four-Step FFT for large power-of-two transforms.
//!
//! Implements Bailey's 4-step algorithm: N = N1 × N2 decomposes the transform
//! into N1 transforms of length N2 and N2 transforms of length N1, interleaved
//! by a twiddle-multiply step using a cached W_N^{j·k} matrix.
//!
//! ## Twiddle caching
//!
//! The W_N^{j·k} matrix (N entries) is built once via a double recurrence
//! (1 cos/sin call total) and reused across all transforms of the same length.
//! This eliminates the O(N) trigonometric evaluations that were previously
//! performed on every call.
//!
//! ## Parallelism
//!
//! Steps 2 and 4 (N1 independent row-FFTs of length N2 and N2 independent
//! row-FFTs of length N1 respectively) are embarrassingly parallel and are
//! executed via Rayon above a configurable threshold.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use rayon::prelude::*;

/// N above which the independent row transforms in steps 2 and 4 use Rayon.
const PARALLEL_ROW_THRESHOLD: usize = 65_536;

/// In-place four-step FFT for large power-of-two lengths.
pub(crate) fn four_step_fft<F: MixedRadixScalar>(data: &mut [F::Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    // Split N = N1 × N2 with N1 ≈ N2 ≈ √N for cache balance.
    let k = n.trailing_zeros();
    let k1 = k / 2;
    let k2 = k - k1;
    let n1 = 1usize << k1; // number of columns / length of second set of FFTs
    let n2 = 1usize << k2; // number of rows / length of first set of FFTs

    let tw1 = if inverse { F::cached_twiddle_inv(n1) } else { F::cached_twiddle_fwd(n1) };
    let tw2 = if inverse { F::cached_twiddle_inv(n2) } else { F::cached_twiddle_fwd(n2) };

    // Cached W_N^{j·k} twiddle matrix, row-major N2 × N1.
    let tw_matrix = F::cached_four_step_twiddles(n, n1, n2, inverse);

    let parallel = n >= PARALLEL_ROW_THRESHOLD;

    F::with_scratch(n, |scratch| {
        // Step 1: transpose data (N1 × N2 logical) → scratch (N2 × N1 layout).
        F::transpose_matrix(data, scratch, n1, n2);

        // Step 2: N1 independent FFTs of length N2 on contiguous rows of scratch.
        // Each row i is scratch[i*n2..(i+1)*n2].  Uses data rows as inner scratch.
        if parallel {
            scratch.par_chunks_exact_mut(n2).for_each(|row| {
                F::with_scratch(n2, |ts| {
                    F::stockham_forward(row, ts, tw2.as_ref());
                });
            });
        } else {
            for (i, row) in scratch.chunks_exact_mut(n2).enumerate() {
                let row_scratch = &mut data[i * n2..(i + 1) * n2];
                F::stockham_forward(row, row_scratch, tw2.as_ref());
            }
        }

        // Step 3: multiply by W_N^{j·k} (cached) and transpose scratch → data.
        // Source layout: scratch[j * n1 + k] for j in 0..n2, k in 0..n1.
        // tw_matrix[j * n1 + k] = W_N^{j·k}.
        // Destination: data[k * n2 + j] (transposed N1 × N2 → N2 × N1 for step 4).
        const TILE: usize = 16;
        for j in (0..n2).step_by(TILE) {
            for kk in (0..n1).step_by(TILE) {
                let j_end = (j + TILE).min(n2);
                let k_end = (kk + TILE).min(n1);
                for r in j..j_end {
                    let src_row = r * n1;
                    for c in kk..k_end {
                        // SAFETY: indices in bounds by loop bounds.
                        let val = unsafe { *scratch.get_unchecked(src_row + c) };
                        let tw = unsafe { *tw_matrix.get_unchecked(src_row + c) };
                        unsafe { *data.get_unchecked_mut(c * n2 + r) = val * tw };
                    }
                }
            }
        }

        // Step 4: N2 independent FFTs of length N1 on contiguous rows of data.
        if parallel {
            data.par_chunks_exact_mut(n1).enumerate().for_each(|(_, row)| {
                F::with_scratch(n1, |ts| {
                    F::stockham_forward(row, ts, tw1.as_ref());
                });
            });
        } else {
            for (i, row) in data.chunks_exact_mut(n1).enumerate() {
                let row_scratch = &mut scratch[i * n1..(i + 1) * n1];
                F::stockham_forward(row, row_scratch, tw1.as_ref());
            }
        }

        // Step 5: transpose data (N2 × N1) → scratch (N1 × N2), copy back.
        F::transpose_matrix(data, scratch, n2, n1);
        data.copy_from_slice(scratch);
    });
}
