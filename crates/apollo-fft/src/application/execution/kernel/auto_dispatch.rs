//! Precision-generic auto-selecting FFT dispatch (SRP: separated from kernel facade).
//!
//! Contains the `FftPrecision` trait, the `fft_precision_impl!` macro that
//! generates implementations from a compact specification, the unified public
//! entry points (`fft_forward`, `fft_inverse`, `fft_inverse_unnorm`), and
//! precision-specific scaffolding (`dispatch_dft11`).

use eunomia::{Complex32, Complex64};

use super::components::winograd::radix::{dft4_array_impl, dft8_array_impl};
use super::components::winograd::{dft3_impl, dft5_array_impl, dft7_impl};
use super::mixed_radix;

/// The cached plan for one length, which every length past the
/// register-resident bases runs through.
///
/// This was briefly conditional. The plan measured slower than
/// `mixed_radix`'s free dispatcher for composite lengths, so the fallback
/// tested `is_power_of_two` and sent composites the other way — a measured
/// guard over a defect rather than a design, recorded with the premise that
/// would retire it. The premise is gone: the two routes were reading their
/// radix decomposition from different tables and disagreeing about the
/// *order*, and the plan's ladder now consults the same static table the
/// dispatcher reads first (`FftPlan1D::new`). With that, the plan is no
/// slower than the free route at any measured length — the composite cells
/// straddle zero across three runs while the power-of-two lengths keep their
/// 12 to 48% win — so there is one route again.
#[inline]
fn cached_plan<F>(n: usize) -> std::sync::Arc<crate::FftPlan1D<F>>
where
    F: crate::application::orchestration::cache::plans::PlanCacheProvider<PlanScalar = F>
        + mixed_radix::MixedRadixScalar,
{
    F::get_1d_plan(
        crate::Shape1D::new(n)
            .expect("invariant: the codelet arms above cover every length below 2"),
    )
}

/// Precision-generic auto-selecting FFT operations.
///
/// Implementors delegate to the `mixed_radix` facade, which routes to:
/// - Stockham autosort for power-of-two lengths (no bit-reversal).
/// - Composite mixed-radix DIT for 2/3/5/7-smooth lengths.
/// - Rader convolution for prime lengths.
///
/// Implemented for `Complex64`, `Complex32`, and `Complex<F16>`.
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
                        cached_plan::<$scalar>(n).forward_complex_slice_inplace(data);
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
                        cached_plan::<$scalar>(n).inverse_complex_slice_inplace(data);
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
                        cached_plan::<$scalar>(n).inverse_complex_slice_unnorm_inplace(data);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
    use crate::application::execution::kernel::test_utils::{max_abs_err_32, max_abs_err_64};
    use eunomia::{Complex, F16};

    #[derive(Clone, Copy)]
    enum CompactTransform {
        Forward,
        Inverse,
        InverseUnnormalized,
    }

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

    fn max_abs_err_half(got: &[Complex<F16>], expected: &[Complex<F16>]) -> f32 {
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

    fn assert_compact_transform_matches_direct(transform: CompactTransform) {
        let n = 96usize;
        let input: Vec<Complex<F16>> = sig32(n)
            .into_iter()
            .map(|value| Complex::new(F16::from_f32(value.re), F16::from_f32(value.im)))
            .collect();
        let promoted: Vec<Complex32> = input
            .iter()
            .map(|value| Complex32::new(value.re.to_f32(), value.im.to_f32()))
            .collect();

        let (expected, transform_scale) = match transform {
            CompactTransform::Forward => (dft_forward(&promoted), 1.0_f32),
            CompactTransform::Inverse => (dft_inverse(&promoted), 1.0 / n as f32),
            CompactTransform::InverseUnnormalized => {
                let mut expected = dft_inverse(&promoted);
                expected.iter_mut().for_each(|value| {
                    value.re *= n as f32;
                    value.im *= n as f32;
                });
                (expected, 1.0)
            }
        };
        let expected: Vec<Complex<F16>> = expected
            .into_iter()
            .map(|value| Complex::new(F16::from_f32(value.re), F16::from_f32(value.im)))
            .collect();

        let input_l1 = promoted
            .iter()
            .map(|value| value.re.abs() + value.im.abs())
            .sum::<f32>();
        let unit_roundoff = F16::EPSILON.to_f32() * 0.5;
        // Each direct output has at most N accumulated terms. The two storage
        // roundings contribute 2u_half; f32 arithmetic contributes at most N*u_f32.
        let compute_roundoff = n as f32 * f32::EPSILON;
        let error_bound = std::f32::consts::SQRT_2
            * input_l1
            * transform_scale
            * (2.0 * unit_roundoff + compute_roundoff);

        let mut actual = input;
        match transform {
            CompactTransform::Forward => fft_forward(&mut actual),
            CompactTransform::Inverse => fft_inverse(&mut actual),
            CompactTransform::InverseUnnormalized => fft_inverse_unnorm(&mut actual),
        }

        let error = max_abs_err_half(&actual, &expected);
        assert!(
            error <= error_bound,
            "compact codelet error {error} exceeds derived bound {error_bound}"
        );
    }

    /// The register-resident storage lengths against a direct DFT.
    ///
    /// `dispatch_compact_storage` routes 2, 4, 8, 16 and 32 over a stack
    /// buffer instead of the pooled `Complex32` scratch, and at some of those
    /// lengths that reaches a different f32 codelet than the general route
    /// would. The lengths the existing compact test covers (96) do not touch
    /// it at all, so the new arms need their own oracle: each length, forward
    /// and inverse, against the direct transform within the same derived
    /// storage bound.
    #[test]
    fn register_resident_storage_lengths_match_direct() {
        for n in [2usize, 4, 8, 16, 32] {
            let input: Vec<Complex<F16>> = sig32(n)
                .into_iter()
                .map(|value| Complex::new(F16::from_f32(value.re), F16::from_f32(value.im)))
                .collect();
            let promoted: Vec<Complex32> = input
                .iter()
                .map(|value| Complex32::new(value.re.to_f32(), value.im.to_f32()))
                .collect();
            let input_l1 = promoted
                .iter()
                .map(|value| value.re.abs() + value.im.abs())
                .sum::<f32>();
            let unit_roundoff = F16::EPSILON.to_f32() * 0.5;
            let compute_roundoff = n as f32 * f32::EPSILON;

            for inverse in [false, true] {
                let expected_f32 = if inverse {
                    dft_inverse(&promoted)
                } else {
                    dft_forward(&promoted)
                };
                let expected: Vec<Complex<F16>> = expected_f32
                    .into_iter()
                    .map(|value| Complex::new(F16::from_f32(value.re), F16::from_f32(value.im)))
                    .collect();
                let scale = if inverse { 1.0 / n as f32 } else { 1.0 };
                let error_bound = std::f32::consts::SQRT_2
                    * input_l1
                    * scale
                    * (2.0 * unit_roundoff + compute_roundoff);

                let mut actual = input.clone();
                if inverse {
                    fft_inverse(&mut actual);
                } else {
                    fft_forward(&mut actual);
                }
                let error = max_abs_err_half(&actual, &expected);
                assert!(
                    error <= error_bound,
                    "n={n} inverse={inverse}: storage error {error} exceeds bound {error_bound}"
                );
            }
        }
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
    fn unified_api_fixed_compact_transforms_match_promoted_direct() {
        assert_compact_transform_matches_direct(CompactTransform::Forward);
        assert_compact_transform_matches_direct(CompactTransform::Inverse);
        assert_compact_transform_matches_direct(CompactTransform::InverseUnnormalized);
    }
}
