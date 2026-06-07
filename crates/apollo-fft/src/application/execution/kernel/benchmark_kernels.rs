//! Precision-generic kernel entry points exposed for benchmarks and tests.
//!
//! Each function is parameterized over `F: MixedRadixScalar + ShortWinogradScalar`,
//! so callers monomorphize at the call site with the desired precision:
//! `benchmark_kernels::rader_prime_forward::<f64>(&mut data)`.
//!
//! Concrete-type wrapper functions (`*_f32` / `*_f64`) are intentionally
//! absent: the type-suffix naming convention is prohibited by the CLAUDE.md
//! canonical-implementation policy.

use super::components::rader::{
    rader_fft, rader_fft_with_convolution_backend, Bluestein, FullCyclic, HalfCyclicWinograd,
};
use super::components::winograd::radix::odd_prime_pair::{dft_pair_impl, PrimePairTable};
use super::components::winograd::WinogradScalar;
use super::mixed_radix::traits::ShortWinogradScalar;
use super::mixed_radix::MixedRadixScalar;
use num_complex::Complex;

/// In-place Rader FFT with automatic convolution strategy and const direction.
pub fn rader_prime_with_direction<F, const INVERSE: bool>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_fft::<F, INVERSE>(data);
}

/// In-place forward Rader FFT with automatic convolution strategy.
pub fn rader_prime_forward<F>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_prime_with_direction::<F, false>(data);
}

/// In-place inverse Rader FFT with automatic convolution strategy.
pub fn rader_prime_inverse<F>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_prime_with_direction::<F, true>(data);
}

/// In-place Rader FFT forced onto the full-cyclic convolution strategy.
pub fn rader_full_cyclic_prime_with_direction<F, const INVERSE: bool>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_fft_with_convolution_backend::<F, INVERSE, FullCyclic>(data);
}

/// In-place forward Rader FFT forced onto the full-cyclic convolution strategy.
pub fn rader_full_cyclic_prime_forward<F>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_full_cyclic_prime_with_direction::<F, false>(data);
}

/// In-place inverse Rader FFT forced onto the full-cyclic convolution strategy.
pub fn rader_full_cyclic_prime_inverse<F>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_full_cyclic_prime_with_direction::<F, true>(data);
}

/// In-place Rader FFT forced onto the half-cyclic Winograd convolution strategy.
pub fn rader_half_cyclic_prime_with_direction<F, const INVERSE: bool>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_fft_with_convolution_backend::<F, INVERSE, HalfCyclicWinograd>(data);
}

/// In-place forward Rader FFT forced onto the half-cyclic Winograd convolution strategy.
pub fn rader_half_cyclic_prime_forward<F>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_half_cyclic_prime_with_direction::<F, false>(data);
}

/// In-place inverse Rader FFT forced onto the half-cyclic Winograd convolution strategy.
pub fn rader_half_cyclic_prime_inverse<F>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_half_cyclic_prime_with_direction::<F, true>(data);
}

/// In-place Rader FFT forced onto the Bluestein convolution strategy.
pub fn rader_bluestein_prime_with_direction<F, const INVERSE: bool>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_fft_with_convolution_backend::<F, INVERSE, Bluestein>(data);
}

/// In-place forward Rader FFT forced onto the Bluestein convolution strategy.
pub fn rader_bluestein_prime_forward<F>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_bluestein_prime_with_direction::<F, false>(data);
}

/// In-place inverse Rader FFT forced onto the Bluestein convolution strategy.
pub fn rader_bluestein_prime_inverse<F>(data: &mut [Complex<F>])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    rader_bluestein_prime_with_direction::<F, true>(data);
}

/// In-place composite mixed-radix forward FFT using an explicit radix sequence.
pub fn composite_forward_with_radices<F>(data: &mut [Complex<F>], radices: &[usize])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortWinogradScalar,
{
    F::composite_forward(data, radices);
}

macro_rules! dispatch_winograd_pair_prime {
    ($F:ty, $data:expr, $inverse:expr, [ $(($p:expr, $h:expr)),* $(,)? ]) => {
        match $data.len() {
            $(
                $p => {
                    dft_pair_impl::<$F, $p, $h, $inverse>(
                        $data.try_into().unwrap(),
                        <$F as PrimePairTable<$p, $h>>::cos_table(),
                        <$F as PrimePairTable<$p, $h>>::sin_table(),
                    );
                }
            )*
            n => panic!("unsupported Winograd-pair benchmark length {n}"),
        }
    };
}

/// In-place Winograd-pair short-prime FFT for canonical lengths with const direction.
pub fn winograd_pair_prime_with_direction<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [Complex<F>],
) {
    dispatch_winograd_pair_prime!(
        F,
        data,
        INVERSE,
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

/// In-place forward Winograd-pair short-prime FFT for canonical lengths.
pub fn winograd_pair_prime_forward<F: WinogradScalar>(data: &mut [Complex<F>]) {
    winograd_pair_prime_with_direction::<F, false>(data);
}

/// In-place inverse Winograd-pair short-prime FFT for canonical lengths.
pub fn winograd_pair_prime_inverse<F: WinogradScalar>(data: &mut [Complex<F>]) {
    winograd_pair_prime_with_direction::<F, true>(data);
}
