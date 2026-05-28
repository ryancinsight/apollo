//! Ordered-layout Rader kernels for fused callers.
//!
//! A standalone natural-order Rader transform must permute nonzero inputs into
//! primitive-root order and permute outputs back to natural frequency order.
//! This module exposes the same convolution core behind an ordered contract:
//! `data[1 + q]` holds `x[g^q]` on input and `X[g_inv^q]` on output. Adjacent
//! fused stages can produce and consume that order directly, eliminating the
//! leaf-local permutation and scratch copy.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

/// Rader transform over an ordered nonzero domain.
///
/// # Contract
/// * `data[0]` is the DC input `x[0]`.
/// * `data[1 + q]` is `x[g^q mod N]` for `q in 0..N-1` on entry.
/// * `data[0]` is `X[0]` on return.
/// * `data[1 + q]` is `X[g_inv^q mod N]` for `q in 0..N-1` on return.
/// Ordered Rader implementation for fused prime paths.
#[inline(always)]
pub(crate) fn rader_ordered_impl<F: MixedRadixScalar<Complex = num_complex::Complex<F>>, const INVERSE: bool>(
    data: &mut [F::Complex],
    n: usize,
    generator_inverse: usize,
) {
    debug_assert!(data.len() >= n);
    debug_assert!(n > 2);
    debug_assert!(crate::application::execution::kernel::radix_shape::is_prime(n));

    let data = &mut data[..n];
    let m = n - 1;
    let (head, nonzero) = data.split_at_mut(1);
    let x0 = head[0];
    let sum_x = sum_ordered::<F>(nonzero);

    if m >= super::BLUESTEIN_RADER_THRESHOLD
        || !crate::application::execution::kernel::radix_shape::is_prime23_smooth(m)
    {
        super::bluestein::rader_bluestein_convolve_inplace::<F>(
            nonzero,
            n,
            INVERSE,
            generator_inverse,
        );
    } else if m >= F::HALF_CYCLIC_RADER_THRESHOLD {
        let half_m = m / 2;
        let (kernel_cyc, kernel_neg) =
            F::cached_rader_negacyclic_spectra(n, INVERSE, generator_inverse);
        let twiddles = F::cached_rader_neg_twiddles(half_m);
        super::rader_negacyclic_convolve_inplace::<F>(
            nonzero,
            kernel_cyc.as_ref(),
            kernel_neg.as_ref(),
            twiddles.as_ref(),
        );
    } else {
        let kernel_spectrum = F::cached_rader_spectrum(n, INVERSE, generator_inverse);
        super::rader_convolve_inplace::<F>(nonzero, kernel_spectrum.as_ref());
    }

    head[0] = x0 + sum_x;
    add_dc_offset::<F>(nonzero, x0);
}

#[inline(always)]
fn sum_ordered<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    nonzero: &[F::Complex],
) -> F::Complex {
    nonzero.iter().copied().sum()
}

#[inline(always)]
fn add_dc_offset<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    nonzero: &mut [F::Complex],
    x0: F::Complex,
) {
    nonzero.iter_mut().for_each(|x| *x = x0 + *x);
}

#[cfg(test)]
mod tests {
    use super::rader_ordered_impl;
    use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
    use num_complex::Complex64;

    fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f64, f64::max)
    }

    fn signal(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let t = k as f64;
                Complex64::new((0.17 * t).sin() + 0.25 * (0.07 * t).cos(), (0.31 * t).cos())
            })
            .collect()
    }

    /// Look up the primitive root and its modular inverse for a prime N
    /// from the canonical [`super::super::generator::PRIMITIVE_ROOTS`] table.
    fn rader_generator_pair(n: usize) -> (usize, usize) {
        super::super::generator::primitive_root_and_inverse(n)
    }

    fn to_ordered_input(input: &[Complex64], n: usize, g: usize) -> Vec<Complex64> {
        let mut ordered = vec![Complex64::new(0.0, 0.0); n];
        ordered[0] = input[0];
        let mut idx = 1usize;
        for q in 0..n - 1 {
            ordered[1 + q] = input[idx];
            idx = (idx * g) % n;
        }
        ordered
    }

    fn to_natural_output(ordered: &[Complex64], n: usize, g_inv: usize) -> Vec<Complex64> {
        let mut natural = vec![Complex64::new(0.0, 0.0); n];
        natural[0] = ordered[0];
        let mut idx = 1usize;
        for q in 0..n - 1 {
            natural[idx] = ordered[1 + q];
            idx = (idx * g_inv) % n;
        }
        natural
    }

    #[test]
    fn ordered_static_forward_matches_direct_for_n29() {
        let (g, g_inv) = rader_generator_pair(29);
        let input = signal(29);
        let expected = dft_forward(&input);
        let mut ordered = to_ordered_input(&input, 29, g);

        rader_ordered_impl::<f64, false>(&mut ordered, 29, g_inv);

        let got = to_natural_output(&ordered, 29, g_inv);
        let err = max_err(&got, &expected);
        assert!(err < 8e-12, "ordered Rader N=29 forward max_err={err:.2e}");
    }

    #[test]
    fn ordered_static_inverse_matches_direct_for_n31() {
        let (g, g_inv) = rader_generator_pair(31);
        let input = signal(31);
        let expected: Vec<_> = dft_inverse(&input).into_iter().map(|x| x * 31.0).collect();
        let mut ordered = to_ordered_input(&input, 31, g);

        rader_ordered_impl::<f64, true>(&mut ordered, 31, g_inv);

        let got = to_natural_output(&ordered, 31, g_inv);
        let err = max_err(&got, &expected);
        assert!(err < 8e-12, "ordered Rader N=31 inverse max_err={err:.2e}");
    }

    #[test]
    fn ordered_runtime_forward_matches_direct_for_n37() {
        let (g, g_inv) = rader_generator_pair(37);
        let input = signal(37);
        let expected = dft_forward(&input);
        let mut ordered = to_ordered_input(&input, 37, g);

        rader_ordered_impl::<f64, false>(&mut ordered, 37, g_inv);

        let got = to_natural_output(&ordered, 37, g_inv);
        let err = max_err(&got, &expected);
        assert!(err < 8e-12, "ordered Rader N=37 forward max_err={err:.2e}");
    }
}
