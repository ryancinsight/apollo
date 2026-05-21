use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

/// Primes handled by two_by_prime dispatch that also appear in the
/// fixed Good-Thomas `short_sizes` list (producing dead-code canonical
/// 2×prime pairs).  Referenced by `fixed::FIXED_EXCLUDE_PRIMES`.
pub(crate) const DIRECT_PAIR_PRIMES: &[usize] = &[11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

apollo_fft_macros::generate_two_by_prime_natural_dispatch! {
    pairs: [
        (11, 5),
        (13, 6),
        (17, 8),
        (19, 9),
        (23, 11),
        (29, 14),
        (31, 15),
        (37, 18),
        (41, 20),
        (43, 21),
        (47, 23),
        (53, 26),
    ],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TwoByPrimeConfig {
    OrderedRader {
        generator: usize,
        generator_inverse: usize,
    },
    NaturalPrime,
}

pub(super) fn try_fft<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
    inverse: bool,
    n1: usize,
    n2: usize,
) -> bool {
    let Some(config) = two_by_prime_n1_config(n1, n2) else {
        return false;
    };

    match config {
        TwoByPrimeConfig::OrderedRader {
            generator,
            generator_inverse,
        } => {
            two_by_prime_ordered_rader::<F>(data, inverse, n1, generator, generator_inverse);
        }
        TwoByPrimeConfig::NaturalPrime => {
            two_by_prime_natural_prime::<F>(data, inverse, n1);
        }
    }
    true
}

fn two_by_prime_n1_config(n1: usize, n2: usize) -> Option<TwoByPrimeConfig> {
    if n2 != 2 || !crate::application::execution::kernel::radix_shape::is_prime(n1) {
        return None;
    }

    if direct_pair_prime(n1) {
        return Some(TwoByPrimeConfig::NaturalPrime);
    }

    if let Some((generator, generator_inverse)) = super::ordered_rader_n1_config(n1) {
        return Some(TwoByPrimeConfig::OrderedRader {
            generator,
            generator_inverse,
        });
    }

    Some(TwoByPrimeConfig::NaturalPrime)
}

#[inline]
pub(super) fn direct_pair_prime(prime: usize) -> bool {
    DIRECT_PAIR_PRIMES.contains(&prime)
}

fn two_by_prime_ordered_rader<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
    inverse: bool,
    prime: usize,
    generator: usize,
    generator_inverse: usize,
) {
    let n = prime * 2;
    debug_assert!(data.len() >= n);

    let twiddles = F::cached_four_step_twiddles(n, prime, 2, inverse);
    let input_order =
        crate::application::execution::kernel::components::rader::cached_generator_order(
            prime, generator,
        );

    F::with_pfa_scratch(n, |scratch| {
        let (even, odd) = scratch[..n].split_at_mut(prime);
        load_even_odd_ordered::<F>(data, even, odd, prime, input_order.as_ref());

        crate::application::execution::kernel::components::rader::ordered::rader_ordered_impl::<F>(
            even,
            inverse,
            prime,
            generator_inverse,
        );
        crate::application::execution::kernel::components::rader::ordered::rader_ordered_impl::<F>(
            odd,
            inverse,
            prime,
            generator_inverse,
        );

        combine_two_prime_ordered::<F>(
            data,
            even,
            odd,
            &twiddles[prime..prime + prime],
            input_order.as_ref(),
            prime,
        );
    });
}

fn two_by_prime_natural_prime<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
    inverse: bool,
    prime: usize,
) {
    let n = prime * 2;
    debug_assert!(data.len() >= n);

    if fuse_two_prime_natural::<F>(data, inverse, prime) {
        return;
    }

    let twiddles = F::cached_four_step_twiddles(n, prime, 2, inverse);
    F::with_pfa_scratch(prime, |scratch| {
        let even = &mut scratch[..prime];
        load_even_compact_odd_natural(data, even, prime);

        transform_natural_prime_half::<F>(even, inverse, prime);
        transform_natural_prime_half::<F>(&mut data[..prime], inverse, prime);

        combine_two_prime_natural_compacted::<F>(
            data,
            even,
            &twiddles[prime..prime + prime],
            prime,
        );
    });
}

#[inline]
fn load_even_odd_ordered<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    src: &[F::Complex],
    even: &mut [F::Complex],
    odd: &mut [F::Complex],
    prime: usize,
    input_order: &[usize],
) {
    debug_assert_eq!(input_order.len(), prime - 1);

    even[0] = src[0];
    odd[0] = src[1];
    for (q, &j) in input_order.iter().enumerate() {
        let src_base = j * 2;
        even[1 + q] = src[src_base];
        odd[1 + q] = src[src_base + 1];
    }
}

#[inline]
fn load_even_compact_odd_natural<C: Copy>(data: &mut [C], even: &mut [C], prime: usize) {
    for j in 0..prime {
        let src_base = j * 2;
        even[j] = data[src_base];
        data[j] = data[src_base + 1];
    }
}

#[inline(always)]
fn transform_natural_prime_half<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
    inverse: bool,
    _prime: usize,
) {
    if inverse {
        crate::application::execution::kernel::mixed_radix::inverse_inplace_unnorm::<F>(data);
    } else {
        crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(data);
    }
}

#[inline]
fn combine_two_prime_ordered<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    dst: &mut [F::Complex],
    even: &[F::Complex],
    odd: &[F::Complex],
    twiddles: &[F::Complex],
    generator_order: &[usize],
    prime: usize,
) {
    debug_assert_eq!(generator_order.len(), prime - 1);
    debug_assert_eq!(twiddles.len(), prime);

    let b0 = odd[0];
    dst[0] = even[0] + b0;
    dst[prime] = even[0] - b0;

    for q in 0..generator_order.len() {
        let k =
            crate::application::execution::kernel::components::rader::inverse_generator_order_at(
                generator_order,
                q,
            );
        let wb = twiddles[k] * odd[1 + q];
        let a = even[1 + q];
        dst[k] = a + wb;
        dst[k + prime] = a - wb;
    }
}

#[inline]
fn combine_two_prime_natural_compacted<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    dst: &mut [F::Complex],
    even: &[F::Complex],
    twiddles: &[F::Complex],
    prime: usize,
) {
    debug_assert_eq!(twiddles.len(), prime);

    for k in 0..prime {
        let wb = twiddles[k] * dst[k];
        let a = even[k];
        dst[k] = a + wb;
        dst[k + prime] = a - wb;
    }
}

#[cfg(test)]
mod tests {
    use super::{two_by_prime_n1_config, TwoByPrimeConfig};

    #[test]
    fn two_by_prime_config_selects_ordered_rader_or_natural_prime() {
        assert_eq!(
            two_by_prime_n1_config(19, 2),
            Some(TwoByPrimeConfig::NaturalPrime)
        );
        assert_eq!(
            two_by_prime_n1_config(29, 2),
            Some(TwoByPrimeConfig::NaturalPrime)
        );
        assert_eq!(
            two_by_prime_n1_config(41, 2),
            Some(TwoByPrimeConfig::NaturalPrime)
        );
        assert_eq!(
            two_by_prime_n1_config(23, 2),
            Some(TwoByPrimeConfig::NaturalPrime)
        );
        assert_eq!(two_by_prime_n1_config(41, 3), None);
        assert_eq!(two_by_prime_n1_config(49, 2), None);
    }
}
