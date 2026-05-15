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

pub mod direct;
pub mod good_thomas;
pub mod mixed_radix;
pub(crate) mod precision_bridge;
pub mod rader;
pub(crate) mod radix_composite;
pub(crate) mod radix_shape;
pub(crate) mod radix_stage;
pub mod real_fft;
pub(crate) mod stockham;
pub(crate) mod tuning;
pub(crate) mod twiddle_table;
pub mod winograd;

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
#[inline]
pub fn fft_forward<C: FftPrecision>(data: &mut [C]) {
    C::fft_forward(data);
}

/// Unified auto-selecting inverse FFT entry point (normalized by 1/N).
#[inline]
pub fn fft_inverse<C: FftPrecision>(data: &mut [C]) {
    C::fft_inverse(data);
}

/// Unified auto-selecting inverse FFT entry point (unnormalized).
#[inline]
pub fn fft_inverse_unnorm<C: FftPrecision>(data: &mut [C]) {
    C::fft_inverse_unnorm(data);
}

// ── FftPrecision implementations ─────────────────────────────────────────────

impl FftPrecision for Complex64 {
    #[inline]
    fn fft_forward(data: &mut [Self]) {
        mixed_radix::forward_inplace::<f64>(data);
    }
    #[inline]
    fn fft_inverse(data: &mut [Self]) {
        mixed_radix::inverse_inplace::<f64>(data);
    }
    #[inline]
    fn fft_inverse_unnorm(data: &mut [Self]) {
        mixed_radix::inverse_inplace_unnorm::<f64>(data);
    }
}

impl FftPrecision for Complex32 {
    #[inline]
    fn fft_forward(data: &mut [Self]) {
        mixed_radix::forward_inplace::<f32>(data);
    }
    #[inline]
    fn fft_inverse(data: &mut [Self]) {
        mixed_radix::inverse_inplace::<f32>(data);
    }
    #[inline]
    fn fft_inverse_unnorm(data: &mut [Self]) {
        mixed_radix::inverse_inplace_unnorm::<f32>(data);
    }
}

impl FftPrecision for Complex<f16> {
    #[inline]
    fn fft_forward(data: &mut [Self]) {
        mixed_radix::forward_compact_storage(data);
    }
    #[inline]
    fn fft_inverse(data: &mut [Self]) {
        mixed_radix::inverse_compact_storage(data);
    }
    #[inline]
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

    fn max_abs_err_f16(got: &[Complex<f16>], expected: &[Complex<f16>]) -> f32 {
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
    fn unified_api_forward_f16_matches_typed() {
        let n = 45usize;
        let input: Vec<Complex<f16>> = sig32(n)
            .into_iter()
            .map(|c| Complex::new(f16::from_f32(c.re), f16::from_f32(c.im)))
            .collect();

        let mut generic = input.clone();
        fft_forward(&mut generic);

        let mut typed = input;
        fft_forward(&mut typed);

        assert!(max_abs_err_f16(&generic, &typed) < 2e-3);
    }
}
