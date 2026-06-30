use apollo_fft_macros::generate_rader_fft;

generate_rader_fft!(5);
generate_rader_fft!(7);
generate_rader_fft!(11);
generate_rader_fft!(13);
generate_rader_fft!(17);
generate_rader_fft!(19);
generate_rader_fft!(23);
generate_rader_fft!(29);
generate_rader_fft!(31);
generate_rader_fft!(37);
generate_rader_fft!(41);
generate_rader_fft!(43);
generate_rader_fft!(47);
generate_rader_fft!(53);

/// Canonical list of primes that have a dedicated static (AST-generated)
/// Rader codelet.  Must stay in sync with the `generate_rader_fft!`
/// invocations above and with the match arms in [`try_static_rader`].
pub(crate) const STATIC_RADER_PRIMES: &[usize] =
    &[5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

#[inline]
pub(crate) fn try_static_rader<F, const INVERSE: bool>(data: &mut [F::Complex], n: usize) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
            Complex = eunomia::Complex<F>,
        > + crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<4>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<6>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<10>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<12>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<16>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<18>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<22>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<28>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<30>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<36>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<40>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<42>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<46>
        + crate::application::execution::kernel::mixed_radix::traits::ShortDft<52>,
{
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
        29 => {
            rader_fft_29::<F, INVERSE>(data);
            true
        }
        31 => {
            rader_fft_31::<F, INVERSE>(data);
            true
        }
        37 => {
            rader_fft_37::<F, INVERSE>(data);
            true
        }
        41 => {
            rader_fft_41::<F, INVERSE>(data);
            true
        }
        43 => {
            rader_fft_43::<F, INVERSE>(data);
            true
        }
        47 => {
            rader_fft_47::<F, INVERSE>(data);
            true
        }
        53 => {
            rader_fft_53::<F, INVERSE>(data);
            true
        }
        _ => false,
    }
}
