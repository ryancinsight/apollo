#[cfg(test)]
use super::traits::WinogradScalar;

pub(crate) mod odd_prime_pair;

// Re-export canonical implementations from butterflies::dft (SSOT for all small DFT kernels).
pub(crate) use crate::application::execution::kernel::components::butterflies::dft::dft4_array_impl;

#[inline]
#[cfg(test)]
pub(crate) fn dft4_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    debug_assert!(data.len() >= 4);
    let data: &mut [num_complex::Complex<F>; 4] =
        (&mut data[..4]).try_into().expect("length checked");
    crate::application::execution::kernel::components::butterflies::dft::dft4_array_impl::<
        F,
        INVERSE,
        false,
    >(data);
}

pub(crate) use crate::application::execution::kernel::components::butterflies::dft::dft8_array_impl;

#[inline]
#[cfg(test)]
pub(crate) fn dft8_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    debug_assert!(data.len() >= 8);
    let data: &mut [num_complex::Complex<F>; 8] =
        (&mut data[..8]).try_into().expect("length checked");
    crate::application::execution::kernel::components::butterflies::dft::dft8_array_impl::<
        F,
        INVERSE,
        false,
    >(data);
}

/// In-place Winograd DFT-7 with optional fused 1/N normalization.
///
/// Re-exported from canonical SSOT in butterflies::dft.
/// Original derivation (Winograd 1978, Blahut 2010 §3.5): 7-point prime DFT
/// exploiting Hermitian twiddle symmetry — 36 real muls, 60 real adds.
pub(crate) use crate::application::execution::kernel::components::butterflies::dft::dft7_impl;

pub(crate) mod dft3;
pub(crate) use dft3::dft3_impl;

/// In-place Good-Thomas DFT-15 (delegate to shared SSOT in butterflies).
/// Only used by tests; runtime dispatch goes through butterflies::dft::dft15_impl.
#[inline]
#[cfg(test)]
pub(crate) fn dft15_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 15],
) {
    crate::application::execution::kernel::components::butterflies::dft::dft15_impl::<F, INVERSE>(
        data,
    );
}

pub(crate) use crate::application::execution::kernel::components::butterflies::dft::dft5_array_impl;

/// In-place DFT-5 (slice form — test ergonomic wrapper).
///
/// Canonical implementation re-exported above; this wrapper casts `&mut [Complex]`
/// to `&mut [Complex; 5]` for test call sites.
#[inline]
#[cfg(test)]
pub(crate) fn dft5_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    debug_assert!(data.len() >= 5);
    let data: &mut [num_complex::Complex<F>; 5] =
        (&mut data[..5]).try_into().expect("length checked");
    crate::application::execution::kernel::components::butterflies::dft::dft5_array_impl::<
        F,
        INVERSE,
        false,
    >(data);
}
