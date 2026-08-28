//! Cache-optimal Four-Step FFT for large power-of-two transforms.
//!
//! Implements Bailey's 4-step algorithm: N = N1 × N2 decomposes the transform
//! into N1 transforms of length N2 and N2 transforms of length N1, interleaved
//! by a twiddle-multiply step using a cached W_N^{j·k} matrix.
//!
//! ## Twiddle caching
//!
//! The W_N^{j·k} matrix (N entries) is evaluated directly once per length and
//! direction, then reused across transforms. Direct evaluation avoids the
//! O(sqrt(N) * u) error growth of the superseded recurrence without putting
//! trigonometric work in the execution path.
//!
//! ## Parallelism
//!
//! Steps 2 and 4 (N1 independent row-FFTs of length N2 and N2 independent
//! row-FFTs of length N1 respectively) are embarrassingly parallel and are
//! executed via Moirai above a configurable threshold.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

/// Tiled in-place square matrix transpose: swaps element (r, c) with (c, r) for r < c.
///
/// Avoids the out-of-place write to scratch followed by `copy_from_slice` that the
/// generic `transpose_matrix` + copy path requires. Cache behaviour: each 16×16 tile
/// pair is loaded into L1 before any writes, so non-sequential strides only appear at
/// the cache-line level, not at the element level.
fn transpose_square_inplace<T: Copy>(data: &mut [T], n: usize) {
    const TILE: usize = 16;
    for i_base in (0..n).step_by(TILE) {
        for j_base in (i_base..n).step_by(TILE) {
            let i_end = (i_base + TILE).min(n);
            let j_end = (j_base + TILE).min(n);
            if i_base == j_base {
                // Diagonal tile: swap strictly upper triangle within the tile.
                for r in i_base..i_end {
                    for c in (r + 1)..j_end {
                        data.swap(r * n + c, c * n + r);
                    }
                }
            } else {
                // Off-diagonal tile: swap with its symmetric mirror tile.
                for r in i_base..i_end {
                    for c in j_base..j_end {
                        data.swap(r * n + c, c * n + r);
                    }
                }
            }
        }
    }
}

/// Crossover for executing independent four-step rows through Moirai.
///
/// This remains separate from the algorithm-selection crossovers: scheduler
/// economics may change without proving that a different transform route wins.
pub(crate) const PARALLEL_ROW_THRESHOLD: usize = 65_536;

/// Runs the shared four-step route when the length admits the selected split.
///
/// Selection lives here so one-dimensional plans and the general mixed-radix
/// dispatcher cannot diverge on split or normalization semantics. The caller
/// supplies its measured workload crossover.
#[inline]
pub(crate) fn try_four_step<
    F: MixedRadixScalar<Complex = eunomia::Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [F::Complex],
    minimum_len: usize,
) -> bool {
    use crate::application::execution::kernel::pot::{FourStep, PotRoute};
    // Admission is the route's own property, defined once on `FourStep`, so the
    // general dispatcher and one-dimensional plans cannot drift apart on which
    // lengths the split is valid for. Only the crossover differs between them,
    // and that is what the caller supplies.
    let n = data.len();
    if n < minimum_len || !FourStep::admits(n) {
        return false;
    }

    FourStep::run::<F, INVERSE, NORMALIZE>(data, &[]);
    true
}

/// In-place four-step FFT for large power-of-two lengths.
/// One radix-2 decimation in time, delegating both halves to the route above.
///
/// `X[k] = E[k] + W_N^k O[k]` and `X[k + N/2] = E[k] - W_N^k O[k]`, with `E`
/// and `O` the transforms of the even- and odd-indexed samples. Both halves
/// are even powers of two, which is exactly the shape the batched planar
/// kernel wants, so an odd power pays one gather, two fast halves, and one
/// combining pass rather than falling to a slower route entirely.
fn radix2_split<F: MixedRadixScalar<Complex = eunomia::Complex<F>>, const INVERSE: bool>(
    data: &mut [F::Complex],
) {
    let n = data.len();
    let half = n / 2;
    let twiddles = if INVERSE {
        F::cached_twiddle_inv(n)
    } else {
        F::cached_twiddle_fwd(n)
    };
    // The stage-major table ends with the length-`n` stage, whose `n / 2`
    // entries are `W_N^j` in order; earlier stages occupy `n / 2 - 1` slots.
    let combine = &twiddles[half - 1..n - 1];

    <F as MixedRadixScalar>::with_scratch(n, |scratch| {
        for (j, pair) in data.chunks_exact(2).enumerate() {
            scratch[j] = pair[0];
            scratch[half + j] = pair[1];
        }
        let (even, odd) = scratch.split_at_mut(half);
        four_step_fft::<F, INVERSE>(even);
        four_step_fft::<F, INVERSE>(odd);

        let (low, high) = data.split_at_mut(half);
        for j in 0..half {
            let rotated = odd[j] * combine[j];
            low[j] = even[j] + rotated;
            high[j] = even[j] - rotated;
        }
    });
}

pub(crate) fn four_step_fft<
    F: MixedRadixScalar<Complex = eunomia::Complex<F>>,
    const INVERSE: bool,
>(
    data: &mut [F::Complex],
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    // The batched layout keeps the transform index in the lane position, which
    // removes every cross-lane shuffle from the butterfly. It covers the square
    // splits below the threading threshold; everything else continues below.
    if F::try_four_step_batched::<INVERSE>(data) {
        return;
    }

    // An odd `log2` has no square split, and the asymmetric one measured
    // badly: the generic path streams a full N-element twiddle matrix, which
    // at N = 8192 cost more than the square route spends on N = 16384. One
    // radix-2 decimation instead leaves two halves that are *even* powers,
    // so each takes the batched planar route above, and the combine reuses
    // the final stage of the existing twiddle table -- `W_N^j` for
    // `j < N/2` already sits at `N/2 - 1`, so no table is added
    // (gap_audit.md#odd-power-routing).
    if n.trailing_zeros() % 2 == 1 && n >= 512 {
        radix2_split::<F, INVERSE>(data);
        return;
    }

    // Split N = N1 × N2 with N1 ≈ N2 ≈ √N for cache balance.
    let k = n.trailing_zeros();
    let k1 = k / 2;
    let k2 = k - k1;
    let n1 = 1usize << k1; // number of columns / length of second set of FFTs
    let n2 = 1usize << k2; // number of rows / length of first set of FFTs

    let tw1 = if INVERSE {
        F::cached_twiddle_inv(n1)
    } else {
        F::cached_twiddle_fwd(n1)
    };
    let tw2 = if INVERSE {
        F::cached_twiddle_inv(n2)
    } else {
        F::cached_twiddle_fwd(n2)
    };

    // Cached W_N^{j·k} twiddle matrix, row-major N2 × N1.
    let tw_matrix = F::cached_four_step_twiddles::<INVERSE>(n, n1, n2);

    let parallel = n >= PARALLEL_ROW_THRESHOLD;

    <F as MixedRadixScalar>::with_scratch(n, |scratch| {
        // Step 1: transpose data (N1 × N2 logical) → scratch (N2 × N1 layout).
        F::transpose_matrix(data, scratch, n1, n2);

        // Step 2: N2 independent FFTs of length N1 on contiguous rows of scratch.
        // After step 1, scratch holds the N2×N1 transposed layout.
        // Each row i is scratch[i*n1..(i+1)*n1].  Uses data rows as inner scratch.
        if parallel {
            moirai::for_each_chunk_mut_with::<moirai::Parallel, _, _>(scratch, n1, |row| {
                <F as MixedRadixScalar>::with_scratch(n1, |ts| {
                    F::stockham_forward(row, ts, tw1.as_ref());
                });
            });
        } else {
            for (i, row) in scratch.chunks_exact_mut(n1).enumerate() {
                let row_scratch = &mut data[i * n1..(i + 1) * n1];
                F::stockham_forward(row, row_scratch, tw1.as_ref());
            }
        }

        // Step 3: multiply by W_N^{j·k} (cached) and transpose scratch → data.
        // Source layout: scratch[j * n1 + k] for j in 0..n2, k in 0..n1.
        // tw_matrix[j * n1 + k] = W_N^{j·k}.
        // Destination: data[k * n2 + j] giving N1 rows of N2 elements for step 4.
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

        // Step 4: N1 independent FFTs of length N2 on contiguous rows of data.
        // After step 3, data holds N1 rows of N2 elements: row k at data[k*n2..].
        if parallel {
            moirai::for_each_chunk_mut_with::<moirai::Parallel, _, _>(data, n2, |row| {
                <F as MixedRadixScalar>::with_scratch(n2, |ts| {
                    F::stockham_forward(row, ts, tw2.as_ref());
                });
            });
        } else {
            for (i, row) in data.chunks_exact_mut(n2).enumerate() {
                let row_scratch = &mut scratch[i * n2..(i + 1) * n2];
                F::stockham_forward(row, row_scratch, tw2.as_ref());
            }
        }

        // Step 5: restore natural-order N1×N2 row-major output.
        // After step 4, data[k1*n2 + k2] = X[k2*n1 + k1] (bit-reversal permuted).
        // A final transpose maps this to data[k2*n1 + k1] = X[k2*n1 + k1].
        // When N1 == N2 (k even) use in-place square transpose.
        if n1 == n2 {
            transpose_square_inplace(data, n1);
        } else {
            F::transpose_matrix(data, scratch, n1, n2);
            data.copy_from_slice(scratch);
        }
    });
}

#[cfg(test)]
mod tests;
