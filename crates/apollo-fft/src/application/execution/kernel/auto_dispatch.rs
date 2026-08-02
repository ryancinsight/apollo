//! Precision-generic auto-selecting FFT dispatch (SRP: separated from kernel facade).
//!
//! Contains the `FftPrecision` trait, the `fft_precision_impl!` macro that
//! generates implementations from a compact specification, the unified public
//! entry points (`fft_forward`, `fft_inverse`, `fft_inverse_unnorm`), and
//! precision-specific scaffolding (`dispatch_dft11`).

use eunomia::{Complex, Complex32, Complex64};
use half::f16;

use super::components::winograd::radix::{dft4_array_impl, dft8_array_impl};
use super::components::winograd::{dft3_impl, dft5_array_impl, dft7_impl};
use super::mixed_radix;

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

/// Macro: generates all three `FftPrecision` methods from one compact specification.
///
/// - `pot_sizes` — power-of-two sizes: dispatched via `small_pot_inplace_sized`
///   (const-generic INVERSE/NORMALIZE handled automatically).
/// - `fn_sizes`   — small-prime sizes: each `ident` must name a function
///   `fn<F: WinogradScalar, const INVERSE: bool>(&mut [Complex<F>; N])`.
///   Inverse-normalized adds a 1/N scale pass via `MixedRadixScalar::complex`.
macro_rules! fft_precision_impl {
    (
        $complex:ty,
        $scalar:ty,
        pot_sizes: [$($pot:literal),* $(,)?],
        fn_sizes: [$($fn_size:literal => $dft_fn:ident),* $(,)?],
    ) => {
        impl FftPrecision for $complex {
            #[inline]
            fn fft_forward(data: &mut [Self]) {
                let n = data.len();
                match n {
                    2 => {
                        let data_ref: &mut [$complex; 2] = data.try_into().unwrap();
                        let a = data_ref[0];
                        let b = data_ref[1];
                        data_ref[0] = a + b;
                        data_ref[1] = a - b;
                    }
                    4 => {
                        let data_ref: &mut [$complex; 4] = data.try_into().unwrap();
                        dft4_array_impl::<$scalar, false, false>(data_ref);
                    }
                    8 => {
                        let data_ref: &mut [$complex; 8] = data.try_into().unwrap();
                        dft8_array_impl::<$scalar, false, false>(data_ref);
                    }
                    $(
                        $pot => unsafe {
                            <$scalar as mixed_radix::MixedRadixScalar>::small_pot_inplace_sized::<$pot, false, false>(data);
                        }
                    )*
                    $(
                        $fn_size => {
                            let data_ref: &mut [$complex; $fn_size] = data.try_into().unwrap();
                            $dft_fn::<$scalar, false>(data_ref);
                        }
                    )*
                    3 => {
                        let data_ref: &mut [$complex; 3] = data.try_into().unwrap();
                        dft3_impl::<$scalar, false, false>(data_ref);
                    }
                    5 => {
                        let data_ref: &mut [$complex; 5] = data.try_into().unwrap();
                        dft5_array_impl::<$scalar, false, false>(data_ref);
                    }
                    7 => {
                        let data_ref: &mut [$complex; 7] = data.try_into().unwrap();
                        dft7_impl::<$scalar, false, false>(data_ref);
                    }
                    _ => {
                        mixed_radix::forward_inplace::<$scalar>(data);
                    }
                }
            }

            #[inline]
            fn fft_inverse(data: &mut [Self]) {
                let n = data.len();
                match n {
                    2 => {
                        let data_ref: &mut [$complex; 2] = data.try_into().unwrap();
                        let a = data_ref[0];
                        let b = data_ref[1];
                        data_ref[0] = <$complex>::new(
                            (a.re + b.re) * 0.5,
                            (a.im + b.im) * 0.5,
                        );
                        data_ref[1] = <$complex>::new(
                            (a.re - b.re) * 0.5,
                            (a.im - b.im) * 0.5,
                        );
                    }
                    4 => {
                        let data_ref: &mut [$complex; 4] = data.try_into().unwrap();
                        dft4_array_impl::<$scalar, true, true>(data_ref);
                    }
                    8 => {
                        let data_ref: &mut [$complex; 8] = data.try_into().unwrap();
                        dft8_array_impl::<$scalar, true, true>(data_ref);
                    }
                    $(
                        $pot => unsafe {
                            <$scalar as mixed_radix::MixedRadixScalar>::small_pot_inplace_sized::<$pot, true, true>(data);
                        }
                    )*
                    $(
                        $fn_size => {
                            let data_ref: &mut [$complex; $fn_size] = data.try_into().unwrap();
                            $dft_fn::<$scalar, true>(data_ref);
                            let scale = <$scalar as mixed_radix::MixedRadixScalar>::complex(
                                1.0 / ($fn_size as f64),
                                0.0,
                            );
                            for x in data_ref.iter_mut() {
                                *x *= scale;
                            }
                        }
                    )*
                    3 => {
                        let data_ref: &mut [$complex; 3] = data.try_into().unwrap();
                        dft3_impl::<$scalar, true, true>(data_ref);
                    }
                    5 => {
                        let data_ref: &mut [$complex; 5] = data.try_into().unwrap();
                        dft5_array_impl::<$scalar, true, true>(data_ref);
                    }
                    7 => {
                        let data_ref: &mut [$complex; 7] = data.try_into().unwrap();
                        dft7_impl::<$scalar, true, true>(data_ref);
                    }
                    _ => {
                        mixed_radix::inverse_inplace::<$scalar>(data);
                    }
                }
            }

            #[inline]
            fn fft_inverse_unnorm(data: &mut [Self]) {
                let n = data.len();
                match n {
                    2 => {
                        let data_ref: &mut [$complex; 2] = data.try_into().unwrap();
                        let a = data_ref[0];
                        let b = data_ref[1];
                        data_ref[0] = a + b;
                        data_ref[1] = a - b;
                    }
                    4 => {
                        let data_ref: &mut [$complex; 4] = data.try_into().unwrap();
                        dft4_array_impl::<$scalar, true, false>(data_ref);
                    }
                    8 => {
                        let data_ref: &mut [$complex; 8] = data.try_into().unwrap();
                        dft8_array_impl::<$scalar, true, false>(data_ref);
                    }
                    $(
                        $pot => unsafe {
                            <$scalar as mixed_radix::MixedRadixScalar>::small_pot_inplace_sized::<$pot, true, false>(data);
                        }
                    )*
                    $(
                        $fn_size => {
                            let data_ref: &mut [$complex; $fn_size] = data.try_into().unwrap();
                            $dft_fn::<$scalar, true>(data_ref);
                        }
                    )*
                    3 => {
                        let data_ref: &mut [$complex; 3] = data.try_into().unwrap();
                        dft3_impl::<$scalar, true, false>(data_ref);
                    }
                    5 => {
                        let data_ref: &mut [$complex; 5] = data.try_into().unwrap();
                        dft5_array_impl::<$scalar, true, false>(data_ref);
                    }
                    7 => {
                        let data_ref: &mut [$complex; 7] = data.try_into().unwrap();
                        dft7_impl::<$scalar, true, false>(data_ref);
                    }
                    _ => {
                        mixed_radix::inverse_inplace_unnorm::<$scalar>(data);
                    }
                }
            }
        }
    };
}

/// Generic wrapper so that the N=11 trait-method call (`ShortWinogradScalar::dft11`)
/// fits the `fn_sizes` convention used by `fft_precision_impl!`.
#[inline]
fn dispatch_dft11<
    F: crate::application::execution::kernel::components::winograd::ShortWinogradScalar,
    const INVERSE: bool,
>(
    data: &mut [eunomia::Complex<F>; 11],
) {
    F::dft11::<INVERSE>(data);
}

fft_precision_impl!(
    Complex64,
    f64,
    pot_sizes: [16, 32, 64],
    fn_sizes: [],
);

fft_precision_impl!(
    Complex32,
    f32,
    pot_sizes: [16, 32, 64],
    fn_sizes: [
        11 => dispatch_dft11,
    ],
);

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
