//! Rader's Algorithm for prime-length FFTs.
//!
//! ## Permutation-fused convolution
//!
//! The standard Rader decomposition rewrites a length-N prime DFT as a
//! length-(N-1) circular convolution.  The classical implementation uses a
//! separate permutation pass: gather `data[g^q mod N]` into an intermediate
//! scratch buffer, then copy that buffer into the zero-padded FFT workspace.
//! Those two passes round-trip through DRAM/L3 for any N where N-1 ≫ L1.
//!
//! This implementation fuses the g^k mod p gather directly into the write of
//! the padded FFT input buffer, so the complex values travel:
//!
//!   data[g^q mod N]  →  padded[q]  →  forward FFT
//!
//! in a single sequential write pass — no intermediate rader_scratch buffer,
//! no copy, no extra cache round-trip.

pub(crate) mod generator;

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// Rader's algorithm for prime N.
///
/// # Precondition
/// `data.len()` must be prime.
pub(crate) fn rader_fft<F: MixedRadixScalar>(data: &mut [F::Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(crate::application::execution::kernel::radix_shape::is_prime(n));

    let g = generator::primitive_root(n);
    let g_inv = mod_inverse(g, n);
    let kernel_spectrum = F::cached_rader_spectrum(n, inverse, g_inv);
    let perm = cached_permutation(n, g, g_inv);

    let x0 = data[0];
    let l = n - 1;
    let m = kernel_spectrum.len();
    let zero = F::complex(0.0, 0.0);

    F::with_rader_padded_scratch(m, |padded| {
        // Fused gather: read data[g^q mod n] directly into padded[q], accumulating
        // the sum needed for data[0].  This eliminates the rader_scratch
        // intermediate buffer and the subsequent copy_from_slice, confining
        // the gather latency to the registers/L1.
        let mut sum_x = zero;
        for (q, &(input_idx, _)) in perm.iter().enumerate() {
            let v = data[input_idx];
            padded[q] = v;
            sum_x = sum_x + v;
        }

        // Zero-fill the linear-convolution guard band padded[l..m].
        // write_bytes maps to a SIMD memset on modern targets.
        // SAFETY: padded[l..] is a valid slice of F::Complex (Copy, no drop glue);
        //         all-zero bytes are a valid representation for Complex<f64/f32>.
        unsafe {
            std::ptr::write_bytes(
                padded[l..].as_mut_ptr().cast::<u8>(),
                0,
                (m - l) * std::mem::size_of::<F::Complex>(),
            );
        }

        // Forward FFT on the zero-padded chirp sequence.
        crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(padded);

        // Pointwise multiply with the pre-computed kernel spectrum.
        F::pointwise_mul(&mut padded[..m], kernel_spectrum.as_ref());

        // Inverse normalized FFT: result is the linear convolution via overlap-add.
        crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(padded);

        // Output scatter with inline overlap-add.
        // padded[q] + padded[q + l] reconstructs the linear convolution output
        // because the zero-padded forward FFT aliased the tail into [l..2l-1].
        data[0] = x0 + sum_x;
        for (q, &(_, output_idx)) in perm.iter().enumerate() {
            let tail = if q + l < m { padded[q + l] } else { zero };
            data[output_idx] = x0 + padded[q] + tail;
        }
    });
}

// ── Permutation cache ─────────────────────────────────────────────────────────

static RADER_PERMUTATION_CACHE: LazyLock<RwLock<HashMap<(usize, usize, usize), Arc<[(usize, usize)]>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Returns cached pairs `(g^q mod n, g_inv^q mod n)` for `q = 0..n-1`.
///
/// The first element is the gather index (input permutation) and the second
/// is the scatter index (output permutation) for the Rader convolution.
fn cached_permutation(n: usize, g: usize, g_inv: usize) -> Arc<[(usize, usize)]> {
    let key = (n, g, g_inv);
    if let Some(permutation) = RADER_PERMUTATION_CACHE.read().get(&key).cloned() {
        return permutation;
    }

    let mut pairs = Vec::with_capacity(n - 1);
    let mut input_idx = 1usize;
    let mut output_idx = 1usize;
    for _ in 0..(n - 1) {
        pairs.push((input_idx, output_idx));
        input_idx = (input_idx * g) % n;
        output_idx = (output_idx * g_inv) % n;
    }
    let permutation: Arc<[(usize, usize)]> = Arc::from(pairs.into_boxed_slice());
    RADER_PERMUTATION_CACHE
        .write()
        .entry(key)
        .or_insert_with(|| Arc::clone(&permutation))
        .clone()
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
