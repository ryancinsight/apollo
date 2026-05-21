//! `ShortWinogradScalar` and generic short-DFT dispatch helpers.
use super::super::radix_stage::normalize_inplace;
use super::super::winograd;
use num_complex::{Complex32, Complex64};

pub(crate) trait ShortWinogradScalar: winograd::WinogradScalar {
    fn dft2(a: &mut num_complex::Complex<Self>, b: &mut num_complex::Complex<Self>);
    fn dft4(data: &mut [num_complex::Complex<Self>; 4], inverse: bool);
    fn dft8(data: &mut [num_complex::Complex<Self>; 8], inverse: bool);
    fn dft16(data: &mut [num_complex::Complex<Self>; 16], inverse: bool);
    fn dft32(data: &mut [num_complex::Complex<Self>; 32], inverse: bool);
    fn dft64(data: &mut [num_complex::Complex<Self>; 64], inverse: bool);
}

impl ShortWinogradScalar for f64 {
    #[inline]
    fn dft2(a: &mut Complex64, b: &mut Complex64) {
        winograd::dft2_impl(a, b);
    }

    #[inline]
    fn dft4(data: &mut [Complex64; 4], inverse: bool) {
        winograd::dft4_impl(data, inverse);
    }

    #[inline]
    fn dft8(data: &mut [Complex64; 8], inverse: bool) {
        winograd::dft8_impl(data, inverse);
    }

    #[inline]
    fn dft16(data: &mut [Complex64; 16], inverse: bool) {
        winograd::dft16_impl(data, inverse);
    }

    #[inline]
    fn dft32(data: &mut [Complex64; 32], inverse: bool) {
        winograd::dft32_impl(data, inverse);
    }

    #[inline]
    fn dft64(data: &mut [Complex64; 64], inverse: bool) {
        winograd::dft64_impl(data, inverse);
    }
}

impl ShortWinogradScalar for f32 {
    #[inline]
    fn dft2(a: &mut Complex32, b: &mut Complex32) {
        winograd::dft2_impl(a, b);
    }

    #[inline]
    fn dft4(data: &mut [Complex32; 4], inverse: bool) {
        winograd::dft4_impl(data, inverse);
    }

    #[inline]
    fn dft8(data: &mut [Complex32; 8], inverse: bool) {
        winograd::dft8_impl(data, inverse);
    }

    #[inline]
    fn dft16(data: &mut [Complex32; 16], inverse: bool) {
        winograd::dft16_impl(data, inverse);
    }

    #[inline]
    fn dft32(data: &mut [Complex32; 32], inverse: bool) {
        winograd::dft32_impl(data, inverse);
    }

    #[inline]
    fn dft64(data: &mut [Complex32; 64], inverse: bool) {
        winograd::dft64_impl(data, inverse);
    }
}

#[inline]
pub(crate) fn forward_short_winograd<F: ShortWinogradScalar>(
    data: &mut [num_complex::Complex<F>],
) -> bool {
    short_winograd(data, false, false)
}

#[inline]
pub(crate) fn inverse_short_winograd<F: ShortWinogradScalar>(
    data: &mut [num_complex::Complex<F>],
    normalize: bool,
) -> bool {
    short_winograd(data, true, normalize)
}

#[inline]
pub(crate) fn short_winograd<F: ShortWinogradScalar>(
    data: &mut [num_complex::Complex<F>],
    inverse: bool,
    normalize: bool,
) -> bool {
    match data.len() {
        2 => {
            let (left, right) = data.split_at_mut(1);
            F::dft2(&mut left[0], &mut right[0]);
        }
        4 => F::dft4(data.try_into().expect("length checked"), inverse),
        8 => F::dft8(data.try_into().expect("length checked"), inverse),
        16 => F::dft16(data.try_into().expect("length checked"), inverse),
        32 => F::dft32(data.try_into().expect("length checked"), inverse),
        64 => F::dft64(data.try_into().expect("length checked"), inverse),
        _ => return false,
    }
    if normalize {
        normalize_inplace(data, F::cast_f64(1.0 / data.len() as f64));
    }
    true
}
