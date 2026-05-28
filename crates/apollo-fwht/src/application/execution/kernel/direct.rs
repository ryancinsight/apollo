//! Walsh-Hadamard transform kernel.
//!
//! ## Mathematical contract
//!
//! The N-point WHT is defined by the Hadamard matrix H_N where
//! `H_N[k,j] = (-1)^{popcount(k & j)}`.
//!
//! H_N is self-inverse up to scaling: H_N^2 = N * I.
//! Normalization: forward is unnormalized; inverse divides by N.
//!
//! ## Involution theorem
//!
//! Theorem: WHT(WHT(x)) = N * x.
//! Proof: H_N^2 = N * I (Hadamard 1893, Walsh 1923 -- H^T * H = N * I, H = H^T).
//! Therefore WHT(WHT(x)) = H_N * (H_N * x) = N * x.

use num_complex::Complex64;
use rayon::prelude::*;
use std::ops::{Add, Sub};

const PAR_THRESHOLD: usize = 1024;

/// Verify that a length is a non-zero power of two.
#[must_use]
pub fn is_valid_length(n: usize) -> bool {
    n > 0 && n.is_power_of_two()
}

/// In-place fast Walsh-Hadamard transform over a slice of type `T`.
///
/// ## Mathematical contract
/// For N = 2^k, the WHT butterfly (a, b) -> (a+b, a-b) is applied at each
/// dyadic scale, yielding O(N log N) operations (Hadamard 1893, Walsh 1923).
///
/// ## Preconditions
/// `data.len()` must be a power of two (enforced by `debug_assert` in debug builds).
/// For release builds, the caller must ensure this invariant; the `FwhtPlan` validates it.
pub fn wht_inplace<T>(data: &mut [T])
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Send + Sync,
{
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "WHT requires power-of-2 length");
    let mut step = 1usize;
    while step < n {
        let block = step * 2;
        if block >= PAR_THRESHOLD {
            data.par_chunks_mut(block).for_each(|chunk| {
                let (left, right) = chunk.split_at_mut(step);
                for i in 0..step {
                    let a = left[i];
                    let b = right[i];
                    left[i] = a + b;
                    right[i] = a - b;
                }
            });
        } else {
            for chunk in data.chunks_mut(block) {
                let (left, right) = chunk.split_at_mut(step);
                for i in 0..step {
                    let a = left[i];
                    let b = right[i];
                    left[i] = a + b;
                    right[i] = a - b;
                }
            }
        }
        step <<= 1;
    }
}

/// In-place Walsh-Hadamard transform over a real slice.
pub fn fwht_inplace(data: &mut [f64]) {
    wht_inplace(data);
}

/// In-place Walsh-Hadamard transform over a complex slice.
pub fn fwht_complex_inplace(data: &mut [Complex64]) {
    wht_inplace(data);
}
