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
    debug_assert_eq!(input_perm.len(), n);
    debug_assert_eq!(output_perm.len(), n);

    F::with_pfa_scratch(n + n1, |scratch| {
        let (matrix, col_buf) = scratch.split_at_mut(n);

        // 1. Gather out-of-place into matrix
        let n4 = (n / 4) * 4;
        let mut j = 0usize;
        while j < n4 {
            unsafe {
                *matrix.get_unchecked_mut(j) = *data.get_unchecked(*input_perm.get_unchecked(j));
                *matrix.get_unchecked_mut(j + 1) = *data.get_unchecked(*input_perm.get_unchecked(j + 1));
                *matrix.get_unchecked_mut(j + 2) = *data.get_unchecked(*input_perm.get_unchecked(j + 2));
                *matrix.get_unchecked_mut(j + 3) = *data.get_unchecked(*input_perm.get_unchecked(j + 3));
            }
            j += 4;
        }
        while j < n {
            unsafe {
                *matrix.get_unchecked_mut(j) = *data.get_unchecked(*input_perm.get_unchecked(j));
            }
            j += 1;
        }

        // 2. Transform rows in-place in matrix
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

        // 3. Transform columns using col_buf, scatter directly to final positions in data
        for i2 in 0..n2 {
            // Extract column - strided access
            unsafe {
                *col_buf.get_unchecked_mut(0) = *matrix.get_unchecked(i2);
                *col_buf.get_unchecked_mut(1) = *matrix.get_unchecked(n2 + i2);
                if n1 > 2 {
                    *col_buf.get_unchecked_mut(2) = *matrix.get_unchecked(2 * n2 + i2);
                }
                if n1 > 3 {
                    *col_buf.get_unchecked_mut(3) = *matrix.get_unchecked(3 * n2 + i2);
                }
                for i1 in 4..n1 {
                    *col_buf.get_unchecked_mut(i1) = *matrix.get_unchecked(i1 * n2 + i2);
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

            // Scatter directly to data using output_perm
            for i1 in 0..n1 {
                unsafe {
                    let out_idx = *output_perm.get_unchecked(i2 * n1 + i1);
                    *data.get_unchecked_mut(out_idx) = *col_buf.get_unchecked(i1);
                }
            }
        }
    });
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
