//! `ShortWinogradScalar` and generic short-DFT dispatch helpers.
use super::super::radix_stage::normalize_inplace;
use super::super::winograd;
use num_complex::{Complex32, Complex64};

pub(crate) trait ShortWinogradScalar: winograd::WinogradScalar {
    fn dft2(a: &mut num_complex::Complex<Self>, b: &mut num_complex::Complex<Self>);
    fn dft3(data: &mut [num_complex::Complex<Self>; 3], inverse: bool);
    fn dft4(data: &mut [num_complex::Complex<Self>; 4], inverse: bool);
    fn dft5(data: &mut [num_complex::Complex<Self>], inverse: bool);
    fn dft6(data: &mut [num_complex::Complex<Self>; 6], inverse: bool);
    fn dft7(data: &mut [num_complex::Complex<Self>; 7], inverse: bool);
    fn dft8(data: &mut [num_complex::Complex<Self>; 8], inverse: bool);
    fn dft9(data: &mut [num_complex::Complex<Self>; 9], inverse: bool);
    fn dft10(data: &mut [num_complex::Complex<Self>; 10], inverse: bool);
    fn dft11(data: &mut [num_complex::Complex<Self>], inverse: bool);
    fn dft12(data: &mut [num_complex::Complex<Self>; 12], inverse: bool);
    fn dft13<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>]);
    fn dft14(data: &mut [num_complex::Complex<Self>; 14], inverse: bool);
    fn dft17<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>]);
    fn dft15(data: &mut [num_complex::Complex<Self>; 15], inverse: bool);
    fn dft16(data: &mut [num_complex::Complex<Self>; 16], inverse: bool);
    fn dft18(data: &mut [num_complex::Complex<Self>; 18], inverse: bool);
    fn dft22(data: &mut [num_complex::Complex<Self>; 22], inverse: bool);
    fn dft23<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>]);
    fn dft25(data: &mut [num_complex::Complex<Self>; 25], inverse: bool);
    fn dft28(data: &mut [num_complex::Complex<Self>; 28], inverse: bool);
    fn dft30(data: &mut [num_complex::Complex<Self>; 30], inverse: bool);
    fn dft32(data: &mut [num_complex::Complex<Self>; 32], inverse: bool);
    fn dft33(data: &mut [num_complex::Complex<Self>; 33], inverse: bool);
    fn dft35(data: &mut [num_complex::Complex<Self>; 35], inverse: bool);
    fn dft36(data: &mut [num_complex::Complex<Self>; 36], inverse: bool);
    fn dft40(data: &mut [num_complex::Complex<Self>; 40], inverse: bool);
    fn dft42(data: &mut [num_complex::Complex<Self>; 42], inverse: bool);
    fn dft45(data: &mut [num_complex::Complex<Self>; 45], inverse: bool);
    fn dft48(data: &mut [num_complex::Complex<Self>; 48], inverse: bool);
    fn dft49(data: &mut [num_complex::Complex<Self>; 49], inverse: bool);
    fn dft50(data: &mut [num_complex::Complex<Self>; 50], inverse: bool);
    fn dft56(data: &mut [num_complex::Complex<Self>; 56], inverse: bool);
    fn dft63(data: &mut [num_complex::Complex<Self>; 63], inverse: bool);
    fn dft64(data: &mut [num_complex::Complex<Self>; 64], inverse: bool);
    fn dft100(data: &mut [num_complex::Complex<Self>; 100], inverse: bool);
}

impl ShortWinogradScalar for f64 {
    #[inline]
    fn dft2(a: &mut Complex64, b: &mut Complex64) {
        winograd::dft2_impl(a, b);
    }

    #[inline]
    fn dft3(data: &mut [Complex64; 3], inverse: bool) {
        winograd::dft3_impl(data, inverse);
    }

    #[inline]
    fn dft4(data: &mut [Complex64; 4], inverse: bool) {
        winograd::dft4_impl(data, inverse);
    }

    #[inline]
    fn dft5(data: &mut [Complex64], inverse: bool) {
        winograd::dft5_impl(data, inverse);
    }

    #[inline]
    fn dft6(data: &mut [Complex64; 6], inverse: bool) {
        winograd::dft6_impl(data, inverse);
    }

    #[inline]
    fn dft7(data: &mut [Complex64; 7], inverse: bool) {
        winograd::dft7_impl(data, inverse);
    }

    #[inline]
    fn dft8(data: &mut [Complex64; 8], inverse: bool) {
        winograd::dft8_impl(data, inverse);
    }

    #[inline]
    fn dft9(data: &mut [Complex64; 9], inverse: bool) {
        winograd::dft9_impl(data, inverse);
    }

    #[inline]
    fn dft10(data: &mut [Complex64; 10], inverse: bool) {
        winograd::dft10_impl(data, inverse);
    }

    #[inline]
    fn dft11(data: &mut [Complex64], inverse: bool) {
        winograd::dft11_impl(data, inverse);
    }

    #[inline]
    fn dft12(data: &mut [Complex64; 12], inverse: bool) {
        winograd::dft12_impl(data, inverse);
    }

    #[inline]
    fn dft13<const INVERSE: bool>(data: &mut [Complex64]) {
        winograd::dft13_impl::<f64, INVERSE>(data);
    }

    #[inline]
    fn dft14(data: &mut [Complex64; 14], inverse: bool) {
        winograd::dft14_impl(data, inverse);
    }

    #[inline]
    fn dft17<const INVERSE: bool>(data: &mut [Complex64]) {
        winograd::dft17_inline_impl::<f64, INVERSE>(data);
    }

    #[inline]
    fn dft15(data: &mut [Complex64; 15], inverse: bool) {
        winograd::dft15_impl(data, inverse);
    }

    #[inline]
    fn dft16(data: &mut [Complex64; 16], inverse: bool) {
        winograd::dft16_impl(data, inverse);
    }

    #[inline]
    fn dft18(data: &mut [Complex64; 18], inverse: bool) {
        winograd::dft18_impl(data, inverse);
    }

    #[inline]
    fn dft22(data: &mut [Complex64; 22], inverse: bool) {
        winograd::dft22_impl(data, inverse);
    }

    #[inline]
    fn dft23<const INVERSE: bool>(data: &mut [Complex64]) {
        winograd::dft23_inline_impl::<f64, INVERSE>(data);
    }

    #[inline]
    fn dft25(data: &mut [Complex64; 25], inverse: bool) {
        winograd::dft25_impl(data, inverse);
    }

    #[inline]
    fn dft28(data: &mut [Complex64; 28], inverse: bool) {
        winograd::dft28_impl(data, inverse);
    }

    #[inline]
    fn dft30(data: &mut [Complex64; 30], inverse: bool) {
        winograd::dft30_impl(data, inverse);
    }

    #[inline]
    fn dft32(data: &mut [Complex64; 32], inverse: bool) {
        winograd::dft32_impl(data, inverse);
    }

    #[inline]
    fn dft33(data: &mut [Complex64; 33], inverse: bool) {
        winograd::dft33_impl(data, inverse);
    }

    #[inline]
    fn dft35(data: &mut [Complex64; 35], inverse: bool) {
        winograd::dft35_impl(data, inverse);
    }

    #[inline]
    fn dft36(data: &mut [Complex64; 36], inverse: bool) {
        winograd::dft36_impl(data, inverse);
    }

    #[inline]
    fn dft40(data: &mut [Complex64; 40], inverse: bool) {
        winograd::dft40_impl(data, inverse);
    }

    #[inline]
    fn dft42(data: &mut [Complex64; 42], inverse: bool) {
        winograd::dft42_impl(data, inverse);
    }

    #[inline]
    fn dft45(data: &mut [Complex64; 45], inverse: bool) {
        winograd::dft45_impl(data, inverse);
    }

    #[inline]
    fn dft48(data: &mut [Complex64; 48], inverse: bool) {
        winograd::dft48_impl(data, inverse);
    }

    #[inline]
    fn dft49(data: &mut [Complex64; 49], inverse: bool) {
        winograd::dft49_impl(data, inverse);
    }

    #[inline]
    fn dft50(data: &mut [Complex64; 50], inverse: bool) {
        winograd::dft50_impl(data, inverse);
    }

    #[inline]
    fn dft56(data: &mut [Complex64; 56], inverse: bool) {
        winograd::dft56_impl(data, inverse);
    }

    #[inline]
    fn dft63(data: &mut [Complex64; 63], inverse: bool) {
        winograd::dft63_impl(data, inverse);
    }

    #[inline]
    fn dft64(data: &mut [Complex64; 64], inverse: bool) {
        winograd::dft64_impl(data, inverse);
    }

    #[inline]
    fn dft100(data: &mut [Complex64; 100], inverse: bool) {
        winograd::dft100_impl(data, inverse);
    }
}

impl ShortWinogradScalar for f32 {
    #[inline]
    fn dft2(a: &mut Complex32, b: &mut Complex32) {
        winograd::dft2_impl(a, b);
    }

    #[inline]
    fn dft3(data: &mut [Complex32; 3], inverse: bool) {
        winograd::dft3_impl(data, inverse);
    }

    #[inline]
    fn dft4(data: &mut [Complex32; 4], inverse: bool) {
        winograd::dft4_impl(data, inverse);
    }

    #[inline]
    fn dft5(data: &mut [Complex32], inverse: bool) {
        winograd::dft5_impl(data, inverse);
    }

    #[inline]
    fn dft6(data: &mut [Complex32; 6], inverse: bool) {
        winograd::dft6_impl(data, inverse);
    }

    #[inline]
    fn dft7(data: &mut [Complex32; 7], inverse: bool) {
        winograd::dft7_impl(data, inverse);
    }

    #[inline]
    fn dft8(data: &mut [Complex32; 8], inverse: bool) {
        winograd::dft8_impl(data, inverse);
    }

    #[inline]
    fn dft9(data: &mut [Complex32; 9], inverse: bool) {
        winograd::dft9_impl(data, inverse);
    }

    #[inline]
    fn dft10(data: &mut [Complex32; 10], inverse: bool) {
        winograd::dft10_impl(data, inverse);
    }

    #[inline]
    fn dft11(data: &mut [Complex32], inverse: bool) {
        winograd::dft11_impl(data, inverse);
    }

    #[inline]
    fn dft12(data: &mut [Complex32; 12], inverse: bool) {
        winograd::dft12_impl(data, inverse);
    }

    #[inline]
    fn dft13<const INVERSE: bool>(data: &mut [Complex32]) {
        winograd::dft13_impl::<f32, INVERSE>(data);
    }

    #[inline]
    fn dft14(data: &mut [Complex32; 14], inverse: bool) {
        winograd::dft14_impl(data, inverse);
    }

    #[inline]
    fn dft17<const INVERSE: bool>(data: &mut [Complex32]) {
        winograd::dft17_impl::<f32, INVERSE>(data);
    }

    #[inline]
    fn dft15(data: &mut [Complex32; 15], inverse: bool) {
        winograd::dft15_impl(data, inverse);
    }

    #[inline]
    fn dft16(data: &mut [Complex32; 16], inverse: bool) {
        winograd::dft16_impl(data, inverse);
    }

    #[inline]
    fn dft18(data: &mut [Complex32; 18], inverse: bool) {
        winograd::dft18_impl(data, inverse);
    }

    #[inline]
    fn dft22(data: &mut [Complex32; 22], inverse: bool) {
        winograd::dft22_impl(data, inverse);
    }

    #[inline]
    fn dft23<const INVERSE: bool>(data: &mut [Complex32]) {
        winograd::dft23_impl::<f32, INVERSE>(data);
    }

    #[inline]
    fn dft25(data: &mut [Complex32; 25], inverse: bool) {
        winograd::dft25_impl(data, inverse);
    }

    #[inline]
    fn dft28(data: &mut [Complex32; 28], inverse: bool) {
        winograd::dft28_impl(data, inverse);
    }

    #[inline]
    fn dft30(data: &mut [Complex32; 30], inverse: bool) {
        winograd::dft30_impl(data, inverse);
    }

    #[inline]
    fn dft32(data: &mut [Complex32; 32], inverse: bool) {
        winograd::dft32_impl(data, inverse);
    }

    #[inline]
    fn dft33(data: &mut [Complex32; 33], inverse: bool) {
        winograd::dft33_impl(data, inverse);
    }

    #[inline]
    fn dft35(data: &mut [Complex32; 35], inverse: bool) {
        winograd::dft35_impl(data, inverse);
    }

    #[inline]
    fn dft36(data: &mut [Complex32; 36], inverse: bool) {
        winograd::dft36_impl(data, inverse);
    }

    #[inline]
    fn dft40(data: &mut [Complex32; 40], inverse: bool) {
        winograd::dft40_impl(data, inverse);
    }

    #[inline]
    fn dft42(data: &mut [Complex32; 42], inverse: bool) {
        winograd::dft42_impl(data, inverse);
    }

    #[inline]
    fn dft45(data: &mut [Complex32; 45], inverse: bool) {
        winograd::dft45_impl(data, inverse);
    }

    #[inline]
    fn dft48(data: &mut [Complex32; 48], inverse: bool) {
        winograd::dft48_impl(data, inverse);
    }

    #[inline]
    fn dft49(data: &mut [Complex32; 49], inverse: bool) {
        winograd::dft49_impl(data, inverse);
    }

    #[inline]
    fn dft50(data: &mut [Complex32; 50], inverse: bool) {
        winograd::dft50_impl(data, inverse);
    }

    #[inline]
    fn dft56(data: &mut [Complex32; 56], inverse: bool) {
        winograd::dft56_impl(data, inverse);
    }

    #[inline]
    fn dft63(data: &mut [Complex32; 63], inverse: bool) {
        winograd::dft63_impl(data, inverse);
    }

    #[inline]
    fn dft64(data: &mut [Complex32; 64], inverse: bool) {
        winograd::dft64_impl(data, inverse);
    }

    #[inline]
    fn dft100(data: &mut [Complex32; 100], inverse: bool) {
        winograd::dft100_impl(data, inverse);
    }
}

#[inline(always)]
pub(crate) fn forward_short_winograd<F: ShortWinogradScalar>(
    data: &mut [num_complex::Complex<F>],
) -> bool {
    short_winograd(data, false, false)
}

#[inline(always)]
pub(crate) fn inverse_short_winograd<F: ShortWinogradScalar>(
    data: &mut [num_complex::Complex<F>],
    normalize: bool,
) -> bool {
    short_winograd(data, true, normalize)
}

#[inline(always)]
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
        3 => F::dft3(data.try_into().expect("length checked"), inverse),
        4 => F::dft4(data.try_into().expect("length checked"), inverse),
        5 => F::dft5(data, inverse),
        6 => F::dft6(data.try_into().expect("length checked"), inverse),
        7 => F::dft7(data.try_into().expect("length checked"), inverse),
        8 => F::dft8(data.try_into().expect("length checked"), inverse),
        9 => F::dft9(data.try_into().expect("length checked"), inverse),
        10 => F::dft10(data.try_into().expect("length checked"), inverse),
        11 => F::dft11(data, inverse),
        12 => F::dft12(data.try_into().expect("length checked"), inverse),
        13 => {
            if inverse {
                F::dft13::<true>(data)
            } else {
                F::dft13::<false>(data)
            }
        }
        14 => F::dft14(data.try_into().expect("length checked"), inverse),
        17 => {
            if inverse {
                F::dft17::<true>(data)
            } else {
                F::dft17::<false>(data)
            }
        }
        15 => F::dft15(data.try_into().expect("length checked"), inverse),
        16 => F::dft16(data.try_into().expect("length checked"), inverse),
        18 => F::dft18(data.try_into().expect("length checked"), inverse),
        22 => F::dft22(data.try_into().expect("length checked"), inverse),
        23 => {
            if inverse {
                F::dft23::<true>(data)
            } else {
                F::dft23::<false>(data)
            }
        }
        25 => F::dft25(data.try_into().expect("length checked"), inverse),
        28 => F::dft28(data.try_into().expect("length checked"), inverse),
        30 => F::dft30(data.try_into().expect("length checked"), inverse),
        32 => F::dft32(data.try_into().expect("length checked"), inverse),
        33 => F::dft33(data.try_into().expect("length checked"), inverse),
        35 => F::dft35(data.try_into().expect("length checked"), inverse),
        36 => F::dft36(data.try_into().expect("length checked"), inverse),
        40 => F::dft40(data.try_into().expect("length checked"), inverse),
        42 => F::dft42(data.try_into().expect("length checked"), inverse),
        45 => F::dft45(data.try_into().expect("length checked"), inverse),
        48 => F::dft48(data.try_into().expect("length checked"), inverse),
        49 => F::dft49(data.try_into().expect("length checked"), inverse),
        50 => F::dft50(data.try_into().expect("length checked"), inverse),
        56 => F::dft56(data.try_into().expect("length checked"), inverse),
        63 => F::dft63(data.try_into().expect("length checked"), inverse),
        64 => F::dft64(data.try_into().expect("length checked"), inverse),
        100 => F::dft100(data.try_into().expect("length checked"), inverse),
        _ => return false,
    }
    if normalize {
        normalize_inplace(data, F::cast_f64(1.0 / data.len() as f64));
    }
    true
}
