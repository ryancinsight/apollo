//! Rader's Algorithm for prime-length FFTs

pub(crate) mod generator;

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

// Rader's algorithm for prime lengths
/// Rader's Algorithm for prime N
pub(crate) fn rader_fft<F: MixedRadixScalar>(data: &mut [F::Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(crate::application::execution::kernel::radix_shape::is_prime(n));

    let g = generator::primitive_root(n);
    let g_inv = mod_inverse(g, n);
    let kernel_spectrum = F::cached_rader_spectrum(n, inverse, g_inv);
    let permutation = cached_permutation(n, g, g_inv);

    let x0 = data[0];

    F::with_rader_scratch(n - 1, |scratch| {
        let mut sum_x = F::complex(0.0, 0.0);
        for (q, &(input_idx, _)) in permutation.iter().enumerate() {
            let value = data[input_idx];
            scratch[q] = value;
            sum_x = sum_x + value;
        }

        circular_convolution_inplace::<F>(scratch, kernel_spectrum.as_ref());

        data[0] = x0 + sum_x;

        for (q, &(_, output_idx)) in permutation.iter().enumerate() {
            data[output_idx] = x0 + scratch[q];
        }
    });
}

static RADER_PERMUTATION_CACHE: LazyLock<RwLock<HashMap<(usize, usize, usize), Arc<[(usize, usize)]>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn cached_permutation(n: usize, g: usize, g_inv: usize) -> Arc<[(usize, usize)]> {
    let key = (n, g, g_inv);
    if let Some(permutation) = RADER_PERMUTATION_CACHE.read().get(&key).cloned() {
        return permutation;
    }

    let mut pairs = Vec::with_capacity(n - 1);
    let mut input_idx = 1;
    let mut output_idx = 1;
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

fn circular_convolution_inplace<F: MixedRadixScalar>(a: &mut [F::Complex], kernel_spectrum: &[F::Complex]) {
    let l = a.len();
    let m = kernel_spectrum.len();
    debug_assert!(m >= 2 * l - 1);

    F::with_rader_padded_scratch(m, |scratch_a| {
        scratch_a[..l].copy_from_slice(a);
        for value in &mut scratch_a[l..m] {
            *value = F::complex(0.0, 0.0);
        }

        crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(scratch_a);
        F::pointwise_mul(&mut scratch_a[..m], kernel_spectrum);
        crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(scratch_a);

        for n in 0..l {
            let tail = if n + l < m {
                scratch_a[n + l]
            } else {
                F::complex(0.0, 0.0)
            };
            a[n] = scratch_a[n] + tail;
        }
    });
}
