use apollo_fft_macros::generate_rader_fft;

generate_rader_fft!(5);
generate_rader_fft!(7);
generate_rader_fft!(11);
generate_rader_fft!(13);
generate_rader_fft!(17);
generate_rader_fft!(19);
generate_rader_fft!(23);
generate_rader_fft!(31);

/// Canonical list of primes that have a dedicated static (AST-generated)
/// Rader codelet.  Must stay in sync with the `generate_rader_fft!`
/// invocations above and with the match arms in [`try_static_rader`].
pub(crate) const STATIC_RADER_PRIMES: &[usize] = &[5, 7, 11, 13, 17, 19, 23, 31];

#[inline(always)]
pub(crate) fn try_static_rader<
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = num_complex::Complex<F>,
    >,
    const INVERSE: bool,
>(
    data: &mut [F::Complex],
    n: usize,
) -> bool {
    // Fast-reject: skip the match if the prime isn't in the canonical set.
    if !STATIC_RADER_PRIMES.contains(&n) {
        return false;
    }
    match n {
        5 => {
            rader_fft_5::<F, INVERSE>(data);
            true
        }
        7 => {
            rader_fft_7::<F, INVERSE>(data);
            true
        }
        11 => {
            rader_fft_11::<F, INVERSE>(data);
            true
        }
        13 => {
            rader_fft_13::<F, INVERSE>(data);
            true
        }
        17 => {
            rader_fft_17::<F, INVERSE>(data);
            true
        }
        19 => {
            rader_fft_19::<F, INVERSE>(data);
            true
        }
        23 => {
            rader_fft_23::<F, INVERSE>(data);
            true
        }
        31 => {
            rader_fft_31::<F, INVERSE>(data);
            true
        }
        _ => false,
    }
}
