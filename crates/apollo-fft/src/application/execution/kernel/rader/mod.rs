//! Rader's Algorithm for prime-length FFTs.
//!
//! ## Circular convolution via direct DFT-{N-1}
//!
//! The standard Rader decomposition rewrites a length-N prime DFT as a
//! length-(N-1) circular convolution.  The precomputed kernel spectrum has
//! length N-1, and the convolution is computed via forward DFT-{N-1},
//! pointwise multiply, and normalized inverse DFT-{N-1}.  The DFT-{N-1} call
//! routes through the composite/PFA path (e.g., DFT-18 = 2×DFT-9 for N=19),
//! which is far shorter than the previous next_power_of_two(2*(N-1)-1) zero-
//! padded path (e.g., DFT-64 for N=19..31).

pub(crate) mod generator;

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use std::sync::Arc;

/// Rader's algorithm for prime N.
///
/// # Precondition
/// `data.len()` must be prime.
pub(crate) fn rader_fft<F: MixedRadixScalar>(data: &mut [F::Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(crate::application::execution::kernel::radix_shape::is_prime(n));

    let g = generator::primitive_root(n);
    let g_inv = mod_inverse(g, n);

    // Forward kernel: H[k] = DFT_{N-1}(W_N^{-g^{-j}})  (sign = -1)
    // Inverse kernel: H[k] = DFT_{N-1}(W_N^{+g^{-j}})  (sign = +1)
    // Both are cached under (n, direction_bit, g_inv).
    let kernel_spectrum = F::cached_rader_spectrum(n, inverse, g_inv);
    // Permutation is direction-independent: gather with g^q, scatter with g_inv^q.
    let (gather, scatter) = cached_permutation(n, g, g_inv);

    let x0 = data[0];
    let l = n - 1;

    // Rader convolution in the thread-local scratch buffer.
    F::with_rader_padded_scratch(l, |padded| {
        // Fused gather + x0 accumulation — single pass over contiguous gather[].
        let mut sum_x = F::complex(0.0, 0.0);
        for (q, &input_idx) in gather.iter().enumerate() {
            let v = data[input_idx];
            padded[q] = v;
            sum_x = sum_x + v;
        }

        rader_convolve_inplace::<F>(padded, kernel_spectrum.as_ref());

        data[0] = x0 + sum_x;
        // Fused scatter + x0 offset — single pass over contiguous scatter[].
        for (q, &output_idx) in scatter.iter().enumerate() {
            data[output_idx] = x0 + padded[q];
        }
    });
}

/// In-place circular convolution via forward FFT → pointwise multiply → inverse FFT.
///
/// `padded` holds the input sequence on entry and the convolution result on exit.
/// `kernel_spectrum` is the precomputed direction-specific DFT of the convolution kernel.
///
/// Named helper: provides a clean insertion point for future stage-level fusion
/// (merging the last forward butterfly with the multiply and first inverse butterfly).
#[inline]
fn rader_convolve_inplace<F: MixedRadixScalar>(
    padded: &mut [F::Complex],
    kernel_spectrum: &[F::Complex],
) {
    // Forward DFT-{N-1}: routes through composite/PFA (no zero-padding).
    crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(padded);
    F::pointwise_mul(padded, kernel_spectrum);
    // Normalized inverse DFT-{N-1}: result is the circular convolution.
    crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(padded);
}

// ── Permutation cache ─────────────────────────────────────────────────────────

/// Returns cached split permutation arrays `(gather_indices, scatter_indices)`.
///
/// `gather[q] = g^q mod n` is the input gather index.
/// `scatter[q] = g_inv^q mod n` is the output scatter index.
///
/// ## Split representation
///
/// Two contiguous `Arc<[usize]>` arrays instead of one interleaved
/// `Arc<[(usize, usize)]>` halves per-element memory footprint in each loop
/// pass and enables independent autovectorization of gather and scatter phases.
/// Direction (forward/inverse) does not change this permutation structure.
fn cached_permutation(n: usize, g: usize, g_inv: usize) -> (Arc<[usize]>, Arc<[usize]>) {
    crate::application::execution::kernel::mixed_radix::caches::cached_rader_perm(
        (n, g, g_inv),
        |(n, g, g_inv)| build_permutation(n, g, g_inv),
    )
}

fn build_permutation(n: usize, g: usize, g_inv: usize) -> (Vec<usize>, Vec<usize>) {
    let l = n - 1;
    let mut gather = Vec::with_capacity(l);
    let mut scatter = Vec::with_capacity(l);
    let mut g_idx = 1usize;
    let mut gi_idx = 1usize;
    for _ in 0..l {
        gather.push(g_idx);
        scatter.push(gi_idx);
        g_idx = (g_idx * g) % n;
        gi_idx = (gi_idx * g_inv) % n;
    }
    (gather, scatter)
}

fn mod_inverse(a: usize, m: usize) -> usize {
    let mut m0 = m as i64;
    let mut y = 0i64;
    let mut x = 1i64;
    let mut a_i64 = a as i64;

    if m == 1 {
        return 0;
    }

    while a_i64 > 1 {
        let q = a_i64 / m0;
        let mut t = m0;
        m0 = a_i64 % m0;
        a_i64 = t;
        t = y;
        y = x - q * y;
        x = t;
    }

    if x < 0 {
        x += m as i64;
    }
    x as usize
}
