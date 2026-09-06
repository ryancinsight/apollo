use super::transpose::transpose_square_inplace;
use super::PARALLEL_ROW_THRESHOLD;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

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
    scratch: &mut [F::Complex],
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

    let (gathered, child_scratch) = scratch.split_at_mut(n);
    for (j, pair) in data.chunks_exact(2).enumerate() {
        gathered[j] = pair[0];
        gathered[half + j] = pair[1];
    }
    let (even, odd) = gathered.split_at_mut(half);
    four_step_fft::<F, INVERSE, false>(even, child_scratch);
    four_step_fft::<F, INVERSE, false>(odd, child_scratch);

    let (low, high) = data.split_at_mut(half);
    for j in 0..half {
        let rotated = odd[j] * combine[j];
        low[j] = even[j] + rotated;
        high[j] = even[j] - rotated;
    }
}

/// Transforms using a caller-owned workspace for the complete decomposition.
///
/// The workspace remains disjoint from `data` and may contain arbitrary values;
/// every used element is written before it is read. Normalization applies once
/// after all decomposition stages.
///
/// # Panics
///
/// Panics before mutating `data` if the length is inadmissible, the workspace
/// requirement is unrepresentable, or `scratch` is too short.
pub(crate) fn four_step_fft<
    F: MixedRadixScalar<Complex = eunomia::Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
) {
    let n = data.len();
    let required = super::scratch_len(n)
        .expect("invariant: four-step length admits a representable workspace");
    assert!(
        scratch.len() >= required,
        "four-step workspace requires {required} elements, received {}",
        scratch.len()
    );
    decompose::<F, INVERSE>(data, &mut scratch[..required]);
    if INVERSE && NORMALIZE {
        F::normalize(data, n);
    }
}

fn decompose<F: MixedRadixScalar<Complex = eunomia::Complex<F>>, const INVERSE: bool>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
) {
    let n = data.len();
    // The batched layout keeps the transform index in the lane position, which
    // removes every cross-lane shuffle from the butterfly. It covers the square
    // splits below the threading threshold; everything else continues below.
    if F::try_four_step_batched::<INVERSE>(data, scratch) {
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
    //
    // This is the unfused form, and it is now the fallback rather than the
    // route: where both halves are themselves planar the call above takes
    // them fused, reading each subsequence out of `data` at stride two
    // instead of gathering it here. What is left for this path is the sizes
    // whose halves exceed the planar threshold and thread their rows.
    if n.trailing_zeros() % 2 == 1 && n >= 512 {
        radix2_split::<F, INVERSE>(data, scratch);
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

    // Step 1: transpose data (N1 × N2 logical) → scratch (N2 × N1 layout).
    F::transpose_matrix(data, scratch, n1, n2);

    // Step 2: N2 independent FFTs of length N1 on contiguous rows of scratch.
    // The corresponding rows in data are inactive and provide Stockham scratch.
    if parallel {
        moirai::for_each_chunk_pair_mut_enumerated_with::<moirai::Parallel, _, _, _>(
            scratch,
            data,
            n1,
            |_, row, row_scratch| {
                F::stockham_forward(row, row_scratch, tw1.as_ref());
            },
        );
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
                    // SAFETY: r < n2 and c < n1 bound src_row + c by n.
                    let val = unsafe { *scratch.get_unchecked(src_row + c) };
                    // SAFETY: the cached matrix has n entries in the same
                    // n2-by-n1 layout as scratch.
                    let tw = unsafe { *tw_matrix.get_unchecked(src_row + c) };
                    // SAFETY: c < n1 and r < n2 bound c*n2 + r by n;
                    // the exclusive data borrow is disjoint from scratch.
                    unsafe { *data.get_unchecked_mut(c * n2 + r) = val * tw };
                }
            }
        }
    }

    // Step 4: N1 independent FFTs of length N2 on contiguous rows of data.
    // The corresponding rows in scratch are inactive and provide Stockham scratch.
    if parallel {
        moirai::for_each_chunk_pair_mut_enumerated_with::<moirai::Parallel, _, _, _>(
            data,
            scratch,
            n2,
            |_, row, row_scratch| {
                F::stockham_forward(row, row_scratch, tw2.as_ref());
            },
        );
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
}
