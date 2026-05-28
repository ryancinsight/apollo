//! Good-Thomas Prime Factor Algorithm (PFA).
//!
//! # Dispatch order (most-specialized first)
//!
//! | Priority | Layer                      | Scope                              |
//! |----------|----------------------------|------------------------------------|
//! | 1        | `two_by_prime`             | (prime, 2) pairs → direct/natural  |
//! | 2        | `three_by_prime`           | (prime, 3) pairs → fused CRT       |
//! | 3        | `cook_toom_gt`             | N=60,84,90,150 hand-fused kernels  |
//! | 4        | `fixed`                    | Generated codelets up to N=200     |
//! | 5        | ordered-Rader PFA          | Primes outside skip set            |
//! | 6        | `pfa_fft_natural_inplace`  | Generic natural-PFA fallback       |
//!
//! Layers 1-4 are statically dispatched via proc-macro-generated
//! `try_fft`/`supports` functions. Layer 5 handles remaining prime
//! factors using Rader-ordered column transforms. Layer 6 is the
//! fully generic fallback with cached CRT permutation cycles.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

mod fixed;
mod three_by_prime;
pub(crate) mod two_by_prime;

// Cook-Toom-GT fused kernels for specific composite sizes
mod cook_toom_gt;

#[inline]
pub(crate) fn has_static_coprime_codelet(n1: usize, n2: usize) -> bool {
    three_by_prime::supports(n1, n2) || fixed::supports(n1, n2)
}

/// Good-Thomas (Prime Factor Algorithm)
///
/// Requires gcd(n1, n2) == 1. Permutation index tables are precomputed and cached
/// on first use via `cached_pfa_perm`, eliminating O(N) modulo per subsequent call.
pub(crate) fn pfa_fft<F: MixedRadixScalar<Complex = num_complex::Complex<F>>, const INVERSE: bool>(
    data: &mut [F::Complex],
    n1: usize,
    n2: usize,
) {
    if two_by_prime::try_fft::<F, INVERSE>(data, n1, n2) {
        return;
    }
    if three_by_prime::try_fft::<F, INVERSE>(data, n1, n2) {
        return;
    }
    // Cook-Toom-GT fused kernel for N=84 (4×21) - checked before fixed
    // since fixed.rs handles 4×21 through the generic Good-Thomas dispatch
    if cook_toom_gt::try_fft::<F, INVERSE>(data, n1, n2) {
        return;
    }
    if fixed::try_fft::<F, INVERSE>(data, n1, n2) {
        return;
    }

    if let Some((generator, generator_inverse)) = ordered_rader_n1_config(n1) {
        pfa_fft_ordered_rader_n1::<F, INVERSE>(data, n1, n2, generator, generator_inverse);
        return;
    }

    pfa_fft_natural_inplace::<F, INVERSE>(data, n1, n2);
}

/// In-place Good-Thomas PFA using precomputed cycle data for efficient permutation.
///
/// Reduces scratch from `n + n1` (gather buffer + column buffer) to just `n1`
/// (column buffer only) by applying the input permutation in-place using
/// precomputed cycle information, eliminating runtime cycle-finding overhead.
///
/// Algorithm:
/// 1. Apply input_perm using precomputed cycles (no runtime cycle detection)
/// 2. Transform rows in-place (contiguous memory - cache-friendly)
/// 3. Transform columns using n1-sized buffer, scatter to final positions
fn pfa_fft_natural_inplace<F: MixedRadixScalar<Complex = num_complex::Complex<F>>, const INVERSE: bool>(
    data: &mut [F::Complex],
    n1: usize,
    n2: usize,
) {
    let n = n1 * n2;
    debug_assert!(data.len() >= n);

    let (input_perm, output_perm) =
        crate::application::execution::kernel::mixed_radix::caches::cached_pfa_perm(n1, n2);
    let cycles =
        crate::application::execution::kernel::mixed_radix::caches::cached_pfa_input_cycles(n1, n2);
    debug_assert_eq!(input_perm.len(), n);
    debug_assert_eq!(output_perm.len(), n);

    // Phase 1: Apply input_perm using precomputed cycles
    // The cycles data contains [len, pos0, pos1, ..., len, pos0, ...]
    // We skip len=1 (fixed points) and rotate longer cycles
    apply_pfa_perm_cycles(data, cycles.as_ref());

    // Phase 2: Transform rows in-place
    // After permutation, rows are contiguous in memory
    for i1 in 0..n1 {
        let row_start = i1 * n2;
        let row_slice = &mut data[row_start..row_start + n2];
        if INVERSE {
            crate::application::execution::kernel::mixed_radix::inverse_inplace_unnorm::<F>(
                row_slice,
            );
        } else {
            crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(row_slice);
        }
    }

    // Phase 3: Transform columns using small buffer, scatter to final positions
    // Only need n1-sized column buffer (not n + n1)
    //
    // The strided column extraction (data[i1 * n2 + i2]) is inherently cache-unfriendly
    // for row-major layout, but modern CPUs handle this well with hardware prefetch.
    // The compiler also auto-vectorizes the strided access when profitable.
    //
    // Optimization: Unrolled extraction for common small n1 values (2, 3, 4) with
    // conditional fallthrough to generic loop for larger n1. This eliminates
    // runtime branch overhead in the hot column extraction loop.
    F::with_pfa_scratch(n1, |col_buf| {
        for i2 in 0..n2 {
            // Extract column - strided access
            // Manual unroll for common small n1 values to eliminate loop overhead
            // The compiler often can't fully unroll due to the strided pattern
            unsafe {
                *col_buf.get_unchecked_mut(0) = *data.get_unchecked(i2);
                *col_buf.get_unchecked_mut(1) = *data.get_unchecked(n2 + i2);
                if n1 > 2 {
                    *col_buf.get_unchecked_mut(2) = *data.get_unchecked(2 * n2 + i2);
                }
                if n1 > 3 {
                    *col_buf.get_unchecked_mut(3) = *data.get_unchecked(3 * n2 + i2);
                }
                // Handle n1 > 4 generically
                for i1 in 4..n1 {
                    *col_buf.get_unchecked_mut(i1) = *data.get_unchecked(i1 * n2 + i2);
                }
            }

            // Transform column
            if INVERSE {
                crate::application::execution::kernel::mixed_radix::inverse_inplace_unnorm::<F>(
                    col_buf,
                );
            } else {
                crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(col_buf);
            }

            // Scatter to final positions in data
            for i1 in 0..n1 {
                unsafe {
                    let out_idx = *output_perm.get_unchecked(i2 * n1 + i1);
                    *data.get_unchecked_mut(out_idx) = *col_buf.get_unchecked(i1);
                }
            }
        }
    });
}

/// Apply PFA input permutation using precomputed cycle data.
///
/// The cycles array contains flat data: [len1, pos1_0, pos1_1, ..., len2, pos2_0, ...]
/// For each cycle with len > 1, we rotate values: values[1..] go to stack[0..len-1],
/// values[0] goes to stack[len-1]. Fixed points (len=1) are skipped.
///
/// This eliminates runtime cycle-finding overhead by precomputing cycles once
/// and caching them. Each call only iterates through the precomputed cycle list.
///
/// Optimization: Stack-allocated temporary for cycles ≤ 32 elements avoids heap
/// allocation. Match arms for common small sizes (2-8) allow LLVM to generate
/// efficient unrolled code. PFA cycles are typically bounded by the smaller
/// coprime factor, so 32 elements covers most real-world cases (512 bytes).
#[inline]
fn apply_pfa_perm_cycles<C: Copy>(data: &mut [C], cycles: &[usize]) {
    let mut idx = 0;
    while idx < cycles.len() {
        let len = cycles[idx];
        idx += 1;

        if len <= 1 {
            idx += len;
            continue;
        }

        let positions = &cycles[idx..idx + len];
        idx += len;

        // Stack-allocated rotation for small cycles (≤32).
        // Match arms for 2-8 allow LLVM to unroll completely.
        // Heap fallback for larger cycles (rare for PFA).
        match len {
            2 => {
                let tmp0 = data[positions[0]];
                let tmp1 = data[positions[1]];
                data[positions[0]] = tmp1;
                data[positions[1]] = tmp0;
            }
            3 => {
                let tmp0 = data[positions[0]];
                let tmp1 = data[positions[1]];
                let tmp2 = data[positions[2]];
                data[positions[0]] = tmp1;
                data[positions[1]] = tmp2;
                data[positions[2]] = tmp0;
            }
            4 => {
                let tmp0 = data[positions[0]];
                let tmp1 = data[positions[1]];
                let tmp2 = data[positions[2]];
                let tmp3 = data[positions[3]];
                data[positions[0]] = tmp1;
                data[positions[1]] = tmp2;
                data[positions[2]] = tmp3;
                data[positions[3]] = tmp0;
            }
            5 => {
                let tmp0 = data[positions[0]];
                let tmp1 = data[positions[1]];
                let tmp2 = data[positions[2]];
                let tmp3 = data[positions[3]];
                let tmp4 = data[positions[4]];
                data[positions[0]] = tmp1;
                data[positions[1]] = tmp2;
                data[positions[2]] = tmp3;
                data[positions[3]] = tmp4;
                data[positions[4]] = tmp0;
            }
            6 => {
                let mut tmp = [data[positions[0]], data[positions[1]], data[positions[2]],
                               data[positions[3]], data[positions[4]], data[positions[5]]];
                data[positions[0]] = tmp[1]; data[positions[1]] = tmp[2];
                data[positions[2]] = tmp[3]; data[positions[3]] = tmp[4];
                data[positions[4]] = tmp[5]; data[positions[5]] = tmp[0];
            }
            7 => {
                let mut tmp = [data[positions[0]], data[positions[1]], data[positions[2]],
                               data[positions[3]], data[positions[4]], data[positions[5]], data[positions[6]]];
                data[positions[0]] = tmp[1]; data[positions[1]] = tmp[2];
                data[positions[2]] = tmp[3]; data[positions[3]] = tmp[4];
                data[positions[4]] = tmp[5]; data[positions[5]] = tmp[6];
                data[positions[6]] = tmp[0];
            }
            8 => {
                let mut tmp = [data[positions[0]], data[positions[1]], data[positions[2]], data[positions[3]],
                               data[positions[4]], data[positions[5]], data[positions[6]], data[positions[7]]];
                data[positions[0]] = tmp[1]; data[positions[1]] = tmp[2];
                data[positions[2]] = tmp[3]; data[positions[3]] = tmp[4];
                data[positions[4]] = tmp[5]; data[positions[5]] = tmp[6];
                data[positions[6]] = tmp[7]; data[positions[7]] = tmp[0];
            }
            _ if len <= 32 => {
                // Stack buffer for 9-32 element cycles (≤512 bytes for Complex64)
                let mut tmp = [data[positions[0]]; 32];
                for i in 0..len { tmp[i] = data[positions[i]]; }
                for k in 0..len {
                    let src = if k + 1 < len { k + 1 } else { 0 };
                    data[positions[k]] = tmp[src];
                }
            }
            _ => {
                // In-place rotation loop for large cycles (>32) to avoid heap allocation
                let first_val = data[positions[0]];
                for k in 0..len - 1 {
                    data[positions[k]] = data[positions[k + 1]];
                }
                data[positions[len - 1]] = first_val;
            }
        }
    }
}

fn pfa_fft_ordered_rader_n1<F: MixedRadixScalar<Complex = num_complex::Complex<F>>, const INVERSE: bool>(
    data: &mut [F::Complex],
    n1: usize,
    n2: usize,
    generator: usize,
    generator_inverse: usize,
) {
    let n = n1 * n2;
    debug_assert!(data.len() >= n);

    let (input_perm, output_perm) =
        crate::application::execution::kernel::mixed_radix::caches::cached_pfa_perm(n1, n2);
    let input_order =
        crate::application::execution::kernel::components::rader::cached_generator_order(
            n1, generator,
        );

    F::with_pfa_scratch(n + n1, |scratch| {
        let (matrix, col_buf) = scratch.split_at_mut(n);

        // 1. Out-of-place gather into `scratch`
        let n4 = (n / 4) * 4;
        let mut j = 0usize;
        while j < n4 {
            unsafe {
                *matrix.get_unchecked_mut(j) = *data.get_unchecked(*input_perm.get_unchecked(j));
                *matrix.get_unchecked_mut(j + 1) =
                    *data.get_unchecked(*input_perm.get_unchecked(j + 1));
                *matrix.get_unchecked_mut(j + 2) =
                    *data.get_unchecked(*input_perm.get_unchecked(j + 2));
                *matrix.get_unchecked_mut(j + 3) =
                    *data.get_unchecked(*input_perm.get_unchecked(j + 3));
            }
            j += 4;
        }
        while j < n {
            unsafe {
                *matrix.get_unchecked_mut(j) = *data.get_unchecked(*input_perm.get_unchecked(j));
            }
            j += 1;
        }

        // 2. Transform rows in `scratch`
        for i1 in 0..n1 {
            let row_start = i1 * n2;
            let row_slice = &mut matrix[row_start..row_start + n2];
            if INVERSE {
                crate::application::execution::kernel::mixed_radix::inverse_inplace_unnorm::<F>(
                    row_slice,
                );
            } else {
                crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(row_slice);
            }
        }

        // 3. Transform cols from `scratch` (with Rader order), output directly to `data`
        for i2 in 0..n2 {
            // Extract column with ordered Rader input mapping
            unsafe {
                *col_buf.get_unchecked_mut(0) = *matrix.get_unchecked(i2);
                for (q, &i1) in input_order.iter().enumerate() {
                    *col_buf.get_unchecked_mut(1 + q) = *matrix.get_unchecked(i1 * n2 + i2);
                }
            }

            // Transform Rader column
            crate::application::execution::kernel::components::rader::ordered::rader_ordered_impl::<
                F, INVERSE
            >(col_buf, n1, generator_inverse);

            // Scatter column directly to final positions in `data`
            unsafe {
                let out_idx_0 = *output_perm.get_unchecked(i2 * n1);
                *data.get_unchecked_mut(out_idx_0) = *col_buf.get_unchecked(0);
                for q in 0..input_order.len() {
                    let k1 =
                        crate::application::execution::kernel::components::rader::inverse_generator_order_at(
                            input_order.as_ref(),
                            q,
                        );
                    let out_idx = *output_perm.get_unchecked(i2 * n1 + k1);
                    *data.get_unchecked_mut(out_idx) = *col_buf.get_unchecked(1 + q);
                }
            }
        }
    });
}

/// Primes that already have dedicated Winograd-pair, static-Rader, or
/// 3×prime dispatch codelets and therefore do not need the generic
/// ordered-Rader PFA path.
///
/// Built from [`three_by_prime::THREE_BY_PRIME_PRIMES`] prefixed with 2
/// and 3. The length (16 = 14 + 2) must be updated if
/// `THREE_BY_PRIME_PRIMES` changes size.
// NOTE: const-block `a.len()` cannot appear in array-size position on
// the current stable toolchain, so the length is hardcoded here.
pub(super) const ORDERED_RADER_SKIP_PRIMES: [usize; 16] = {
    let a = three_by_prime::THREE_BY_PRIME_PRIMES;
    let mut merged = [0usize; 16];
    merged[0] = 2;
    merged[1] = 3;
    let mut i = 0;
    while i < a.len() {
        merged[2 + i] = a[i];
        i += 1;
    }
    merged
};

fn ordered_rader_n1_config(n1: usize) -> Option<(usize, usize)> {
    if ORDERED_RADER_SKIP_PRIMES.contains(&n1) {
        return None;
    }
    if !crate::application::execution::kernel::radix_shape::is_prime(n1) {
        return None;
    }
    Some(crate::application::execution::kernel::components::rader::generator::primitive_root_and_inverse(n1))
}

#[cfg(test)]
mod tests;
