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

/// The `#[inline]` hint here is load-bearing. Do not remove it, and do not
/// replace it with `#[inline(never)]` — that was tried and measured, and it
/// regresses.
///
/// ## What looks wrong but is not
///
/// LLVM's cost model reaches opposite conclusions for the two scalar types
/// under this hint. It declines for `f64`, leaving `rader_fft::<f64>` at 1215
/// instructions with zero stack spills and a 40-byte frame. It accepts for
/// `f32` — whose individual codelets are cheaper, so each looks affordable —
/// and the aggregate exceeds the register file: `rader_fft::<f32>` reaches
/// 4275 instructions with 469 stack-relative moves and a 720-byte frame.
///
/// That asymmetry is real and reads like a defect. Marking this function
/// `#[inline(never)]` does remove it completely: both instantiations then sit
/// near 210 instructions with zero spills and 40-byte frames.
///
/// ## Why it was reverted anyway
///
/// The replicated counterbalanced benchmark gate rejected that change, slower
/// in all four comparisons on every affected row:
///
/// | case | with `#[inline]` | with `#[inline(never)]` |
/// | --- | ---: | ---: |
/// | `rader_f32/53` | 366 ns | 458 ns (+25%) |
/// | `rader_f64/53` | 375 ns | 444 ns (+18%) |
/// | `rader_f64/19` | 106 ns | 115 ns (+8%) |
/// | `half_cyclic_f64/67` | 708 ns | 723 ns (+2%) |
///
/// Inlining lets the caller specialise the static codelets against a known
/// `n`; that specialisation is worth more than the spills it costs. The `f64`
/// rows are the decisive disproof — `f64` was spilling zero times to begin
/// with, so removing the hint could only add call overhead and lost
/// specialisation, and it measurably did.
///
/// The lesson for anyone re-reading the assembly: spill count is a model of
/// cost, not a measurement of it. Route any change here through the
/// counterbalanced gate rather than a static read of the emitted code.
#[inline]
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
