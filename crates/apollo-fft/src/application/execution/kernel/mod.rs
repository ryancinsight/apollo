//! Apollo FFT kernel module.
//!
//! ## Kernel implementations
//!
//! | Module            | Role |
//! |-------------------|------|
//! | `direct`          | O(N²) reference DFT; used only for testing. |
//! | `radix2`          | Twiddle-table builders used by Stockham, Rader, and tests. |
//! | `winograd`        | Short-DFT codelets (DFT-3/5/7/8/N) used by the composite kernel. |
//! | `radix_composite` | Mixed-radix Stockham autosort FFT for 2/3/5/7-smooth composite lengths. |
//! | `stockham`        | Radix-2 Stockham autosort FFT for all power-of-two lengths. |
//! | `mixed_radix`     | Dispatch facade: Stockham for PoT, composite/PFA for smooth, Rader for primes. |

pub(crate) mod components;
pub mod direct;
pub mod mixed_radix;
pub(crate) mod precision_bridge;
pub(crate) mod radix_shape;
pub(crate) mod radix_stage;
pub mod real_fft;
pub(crate) mod tuning;
pub(crate) mod twiddle_table;

#[cfg(any(test, debug_assertions, feature = "kernel-strategy-bench"))]
#[doc(hidden)]
pub mod benchmark_kernels {
    //! Precision-generic kernel entry points exposed for benchmarks and tests.
    //!
    //! Each function is parameterized over `F: MixedRadixScalar + ShortWinogradScalar`,
    //! so callers monomorphize at the call site with the desired precision:
    //! `benchmark_kernels::rader_prime::<f64>(&mut data, inverse)`.
    //!
    //! Concrete-type wrapper functions (`*_f32` / `*_f64`) are intentionally
    //! absent: the type-suffix naming convention is prohibited by the CLAUDE.md
    //! canonical-implementation policy.

    use super::components::rader::{
        rader_fft, rader_fft_with_convolution_strategy, RaderConvolutionStrategy,
    };
    use super::components::winograd::radix::odd_prime_pair::{dft_pair_impl, PrimePairTable};
    use super::components::winograd::WinogradScalar;
    use super::mixed_radix::traits::ShortWinogradScalar;
    use super::mixed_radix::MixedRadixScalar;
    use num_complex::Complex;

    /// In-place forward/inverse Rader FFT with automatic convolution strategy.
    pub fn rader_prime<F>(data: &mut [Complex<F>], inverse: bool)
    where
        F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
    {
        if inverse {
            rader_fft::<F, true>(data);
        } else {
            rader_fft::<F, false>(data);
        }
    }

    /// In-place Rader FFT forced onto the full-cyclic convolution strategy.
    pub fn rader_full_cyclic_prime<F>(data: &mut [Complex<F>], inverse: bool)
    where
        F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
    {
        let strategy = RaderConvolutionStrategy::FullCyclic;
        if inverse {
            rader_fft_with_convolution_strategy::<F, true>(data, strategy);
        } else {
            rader_fft_with_convolution_strategy::<F, false>(data, strategy);
        }
    }

    /// In-place Rader FFT forced onto the half-cyclic Winograd convolution strategy.
    pub fn rader_half_cyclic_prime<F>(data: &mut [Complex<F>], inverse: bool)
    where
        F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
    {
        let strategy = RaderConvolutionStrategy::HalfCyclicWinograd;
        if inverse {
            rader_fft_with_convolution_strategy::<F, true>(data, strategy);
        } else {
            rader_fft_with_convolution_strategy::<F, false>(data, strategy);
        }
    }

    macro_rules! dispatch_winograd_pair_prime {
        ($F:ty, $data:expr, $inverse:expr, [ $(($p:expr, $h:expr)),* $(,)? ]) => {
            match ($data.len(), $inverse) {
                $(
                    ($p, true) => {
                        dft_pair_impl::<$F, $p, $h, true>(
                            $data.try_into().unwrap(),
                            <$F as PrimePairTable<$p, $h>>::cos_table(),
                            <$F as PrimePairTable<$p, $h>>::sin_table(),
                        );
                    }
                    ($p, false) => {
                        dft_pair_impl::<$F, $p, $h, false>(
                            $data.try_into().unwrap(),
                            <$F as PrimePairTable<$p, $h>>::cos_table(),
                            <$F as PrimePairTable<$p, $h>>::sin_table(),
                        );
                    }
                )*
                (n, _) => panic!("unsupported Winograd-pair benchmark length {n}"),
            }
        };
    }

    /// In-place forward/inverse Winograd-pair short-prime FFT for canonical lengths.
    pub fn winograd_pair_prime<F: WinogradScalar>(data: &mut [Complex<F>], inverse: bool) {
        dispatch_winograd_pair_prime!(
            F,
            data,
            inverse,
            [
                (19, 9),
                (29, 14),
                (31, 15),
                (37, 18),
                (41, 20),
                (43, 21),
                (47, 23),
                (53, 26)
            ]
        );
    }
}

#[cfg(test)]
pub(crate) mod test_utils;

pub use direct::{dft_forward, dft_inverse, KernelScalar};

use half::f16;
use num_complex::{Complex, Complex32, Complex64};

// ── Precision-generic auto-selecting API ─────────────────────────────────────

/// Precision-generic auto-selecting FFT operations.
///
/// Implementors delegate to the `mixed_radix` facade, which routes to:
/// - Stockham autosort for power-of-two lengths (no bit-reversal).
/// - Composite mixed-radix DIT for 2/3/5/7-smooth lengths.
/// - Rader convolution for prime lengths.
///
/// Implemented for `Complex64`, `Complex32`, and `Complex<f16>`.
pub trait FftPrecision: Sized {
    /// In-place forward FFT, unnormalized.
    fn fft_forward(data: &mut [Self]);
    /// In-place inverse FFT, normalized by 1/N.
    fn fft_inverse(data: &mut [Self]);
    /// In-place inverse FFT, unnormalized (no 1/N division).
    ///
    /// Use this when normalization is deferred to a single outer call
    /// (e.g., separable multi-dimensional transforms).
    fn fft_inverse_unnorm(data: &mut [Self]);
}

/// Unified auto-selecting forward FFT entry point across all supported precisions.
#[inline(always)]
pub fn fft_forward<C: FftPrecision>(data: &mut [C]) {
    C::fft_forward(data);
}

/// Unified auto-selecting inverse FFT entry point (normalized by 1/N).
#[inline(always)]
pub fn fft_inverse<C: FftPrecision>(data: &mut [C]) {
    C::fft_inverse(data);
}

/// Unified auto-selecting inverse FFT entry point (unnormalized).
#[inline(always)]
pub fn fft_inverse_unnorm<C: FftPrecision>(data: &mut [C]) {
    C::fft_inverse_unnorm(data);
}

// ── FftPrecision implementations ─────────────────────────────────────────────

impl FftPrecision for Complex64 {
    #[inline(always)]
    fn fft_forward(data: &mut [Self]) {
        mixed_radix::forward_inplace::<f64>(data);
    }
    #[inline(always)]
    fn fft_inverse(data: &mut [Self]) {
        mixed_radix::inverse_inplace::<f64>(data);
    }
    #[inline(always)]
    fn fft_inverse_unnorm(data: &mut [Self]) {
        mixed_radix::inverse_inplace_unnorm::<f64>(data);
    }
}

impl FftPrecision for Complex32 {
    #[inline(always)]
    fn fft_forward(data: &mut [Self]) {
        mixed_radix::forward_inplace::<f32>(data);
    }
    #[inline(always)]
    fn fft_inverse(data: &mut [Self]) {
        mixed_radix::inverse_inplace::<f32>(data);
    }
    #[inline(always)]
    fn fft_inverse_unnorm(data: &mut [Self]) {
        mixed_radix::inverse_inplace_unnorm::<f32>(data);
    }
}

impl FftPrecision for Complex<f16> {
    #[inline(always)]
    fn fft_forward(data: &mut [Self]) {
        mixed_radix::forward_compact_storage(data);
    }
    #[inline(always)]
    fn fft_inverse(data: &mut [Self]) {
        mixed_radix::inverse_compact_storage(data);
    }
    #[inline(always)]
    fn fft_inverse_unnorm(data: &mut [Self]) {
        mixed_radix::inverse_unnorm_compact_storage(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::dft_forward;
    use crate::application::execution::kernel::test_utils::{max_abs_err_32, max_abs_err_64};

    fn sig64(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let t = k as f64;
                Complex64::new((0.27 * t).sin(), 0.35 * (0.11 * t).cos())
            })
            .collect()
    }

    fn sig32(n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|k| {
                let t = k as f32;
                Complex32::new((0.27_f32 * t).sin(), 0.35_f32 * (0.11_f32 * t).cos())
            })
            .collect()
    }

    fn max_abs_err_half(got: &[Complex<f16>], expected: &[Complex<f16>]) -> f32 {
        got.iter()
            .zip(expected.iter())
            .map(|(x, y)| {
                let (xr, xi) = (x.re.to_f32(), x.im.to_f32());
                let (yr, yi) = (y.re.to_f32(), y.im.to_f32());
                let dr = xr - yr;
                let di = xi - yi;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn unified_api_forward_64_matches_direct_and_typed() {
        let n = 45usize;
        let input = sig64(n);

        let mut generic = input.clone();
        fft_forward(&mut generic);

        let direct = dft_forward(&input);
        assert!(max_abs_err_64(&generic, &direct) < 1e-10);
    }

    #[test]
    fn unified_api_forward_32_matches_direct_and_typed() {
        let n = 45usize;
        let input = sig32(n);

        let mut generic = input.clone();
        fft_forward(&mut generic);

        let direct = dft_forward(&input);
        assert!(max_abs_err_32(&generic, &direct) < 5e-4);
    }

    #[test]
    fn unified_api_forward_half_matches_typed() {
        let n = 45usize;
        let input: Vec<Complex<f16>> = sig32(n)
            .into_iter()
            .map(|c| Complex::new(f16::from_f32(c.re), f16::from_f32(c.im)))
            .collect();

        let mut generic = input.clone();
        fft_forward(&mut generic);

        let mut typed = input;
        fft_forward(&mut typed);

        assert!(max_abs_err_half(&generic, &typed) < 2e-3);
    }
}
