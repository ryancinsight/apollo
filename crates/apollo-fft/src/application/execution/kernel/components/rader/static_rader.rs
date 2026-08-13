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

const STATIC_RADER_MAX_PRIME: usize = 53;

/// Canonical list of primes that have a dedicated static (AST-generated)
/// Rader codelet.  Must stay in sync with the `generate_rader_fft!`
/// invocations above and with the match arms in [`try_static_rader`].
#[cfg(test)]
pub(crate) const STATIC_RADER_PRIMES: &[usize] = &[
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    STATIC_RADER_MAX_PRIME,
];

/// Kept out of line deliberately: this function body is the whole static
/// codelet table, and inlining it into [`rader_fft`] amplifies rather than
/// helps.
///
/// LLVM's cost model reached opposite conclusions for the two scalar types
/// under a plain `#[inline]` hint. It declined for `f64`, leaving
/// `rader_fft::<f64>` at 1215 instructions with **zero** stack spills and a
/// 40-byte frame. It accepted for `f32` — whose individual codelets are
/// cheaper, so each looks affordable — and the aggregate blew the register
/// file: `rader_fft::<f32>` reached 4275 instructions with **469**
/// stack-relative moves and a 720-byte frame. The measured consequence was
/// `f32` running 1.71x slower than `f64` at N=19 and 1.19x at N=31, despite
/// moving half the bytes per element.
///
/// One call per Rader invocation is a few cycles; several hundred spill
/// round-trips is not. Verify with a spill count over the emitted assembly
/// rather than a timing run — it is exact and immune to host load.
///
/// [`rader_fft`]: super::rader_fft
#[inline(never)]
pub(crate) fn try_static_rader<F, const INVERSE: bool>(data: &mut [F::Complex], n: usize) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
            Complex = eunomia::Complex<F>,
        > + crate::application::execution::kernel::components::winograd::ShortWinogradScalar
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
    // Larger primes always use the dynamic path. Reject them before the
    // generated match so fallback cost does not scale with static codelet count.
    if n > STATIC_RADER_MAX_PRIME {
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
