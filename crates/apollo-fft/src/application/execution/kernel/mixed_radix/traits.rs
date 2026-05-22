//! `ShortWinogradScalar` and generic short-DFT dispatch helpers.

use super::super::components::winograd;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use num_complex::{Complex32, Complex64};

macro_rules! impl_short_winograd_prime_pair {
    ($ty:ty, $(($method:ident, $n:expr, $h:expr)),+ $(,)?) => {
        $(
            #[inline(always)]
            fn $method<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; $n]) {
                use winograd::radix::odd_prime_pair::{dft_pair_impl, PrimePairTable};
                dft_pair_impl::<$ty, $n, $h, INVERSE>(
                    data,
                    <$ty as PrimePairTable<$n, $h>>::cos_table(),
                    <$ty as PrimePairTable<$n, $h>>::sin_table(),
                );
            }
        )+
    };
}

macro_rules! impl_short_winograd_prime_pair_reduced {
    ($(($method:ident, $n:expr, $h:expr)),+ $(,)?) => {
        $(
            #[inline(always)]
            fn $method<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; $n]) {
                use winograd::radix::odd_prime_pair::{dft_pair_impl_reduced, PrimePairTable};
                dft_pair_impl_reduced::<f32, $n, $h, INVERSE>(
                    data,
                    <f32 as PrimePairTable<$n, $h>>::cos_table(),
                    <f32 as PrimePairTable<$n, $h>>::sin_table(),
                );
            }
        )+
    };
}

pub trait ShortWinogradScalar: winograd::WinogradScalar {
    fn dft2(data: &mut [num_complex::Complex<Self>; 2]);
    fn dft3<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 3]);
    fn dft4<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 4]);
    fn dft5<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 5]);
    fn dft7<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 7]);
    fn dft8<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 8]);
    fn dft16<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 16]);
    fn dft11<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 11]);
    fn dft13<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 13]);
    fn dft17<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 17]);
    fn dft19<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 19]);
    fn dft23<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 23]);
    fn dft29<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 29]);
    fn dft31<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 31]);
    fn dft37<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 37]);
    fn dft41<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 41]);
    fn dft43<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 43]);
    fn dft47<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 47]);
    fn dft53<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; 53]);
}

impl ShortWinogradScalar for f64 {
    #[inline(always)]
    fn dft2(data: &mut [Complex64; 2]) {
        winograd::dft2_impl(data);
    }

    #[inline(always)]
    fn dft3<const INVERSE: bool>(data: &mut [Complex64; 3]) {
        winograd::dft3_impl::<f64, INVERSE>(data);
    }

    #[inline(always)]
    fn dft4<const INVERSE: bool>(data: &mut [Complex64; 4]) {
        winograd::dft4_array_impl::<f64, INVERSE>(data);
    }

    #[inline(always)]
    fn dft5<const INVERSE: bool>(data: &mut [Complex64; 5]) {
        winograd::dft5_array_impl::<f64, INVERSE>(data);
    }

    #[inline(always)]
    fn dft7<const INVERSE: bool>(data: &mut [Complex64; 7]) {
        winograd::dft7_impl::<f64, INVERSE>(data);
    }

    #[inline(always)]
    fn dft8<const INVERSE: bool>(data: &mut [Complex64; 8]) {
        winograd::dft8_array_impl::<f64, INVERSE>(data);
    }

    #[inline(always)]
    fn dft16<const INVERSE: bool>(data: &mut [Complex64; 16]) {
        winograd::dft16_impl::<f64, INVERSE>(data);
    }

    impl_short_winograd_prime_pair!(
        f64,
        (dft11, 11, 5),
        (dft13, 13, 6),
        (dft17, 17, 8),
        (dft19, 19, 9),
        (dft23, 23, 11),
        (dft29, 29, 14),
        (dft31, 31, 15),
        (dft37, 37, 18),
        (dft41, 41, 20),
        (dft43, 43, 21),
        (dft47, 47, 23),
        (dft53, 53, 26),
    );
}

impl ShortWinogradScalar for f32 {
    #[inline(always)]
    fn dft2(data: &mut [Complex32; 2]) {
        winograd::dft2_impl(data);
    }

    #[inline(always)]
    fn dft3<const INVERSE: bool>(data: &mut [Complex32; 3]) {
        winograd::dft3_impl::<f32, INVERSE>(data);
    }

    #[inline(always)]
    fn dft4<const INVERSE: bool>(data: &mut [Complex32; 4]) {
        winograd::dft4_array_impl::<f32, INVERSE>(data);
    }

    #[inline(always)]
    fn dft5<const INVERSE: bool>(data: &mut [Complex32; 5]) {
        winograd::dft5_array_impl::<f32, INVERSE>(data);
    }

    #[inline(always)]
    fn dft7<const INVERSE: bool>(data: &mut [Complex32; 7]) {
        winograd::dft7_impl::<f32, INVERSE>(data);
    }

    #[inline(always)]
    fn dft8<const INVERSE: bool>(data: &mut [Complex32; 8]) {
        winograd::dft8_array_impl::<f32, INVERSE>(data);
    }

    #[inline(always)]
    fn dft16<const INVERSE: bool>(data: &mut [Complex32; 16]) {
        winograd::dft16_impl::<f32, INVERSE>(data);
    }

    impl_short_winograd_prime_pair!(
        f32,
        (dft11, 11, 5),
        (dft13, 13, 6),
        (dft17, 17, 8),
        (dft19, 19, 9),
        (dft23, 23, 11),
        (dft29, 29, 14),
        (dft37, 37, 18),
        (dft41, 41, 20),
        (dft43, 43, 21),
        (dft47, 47, 23),
        (dft53, 53, 26),
    );

    impl_short_winograd_prime_pair_reduced!((dft31, 31, 15));
}

/// Canonical catalog of sizes with dedicated Winograd short-DFT codelets.
/// Each entry corresponds to a `ShortDft<N>` impl and a `short_winograd`
/// match arm. Adding or removing a codelet size must update this array.
/// The array **must remain sorted** (ascending) for `is_short_winograd_size`.
pub(crate) const SHORT_WINOGRAD_SIZES: &[usize] = &[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 58, 60, 62, 63, 64, 81, 128,
];

/// O(log N) membership test for `SHORT_WINOGRAD_SIZES` via binary search.
///
/// `SHORT_WINOGRAD_SIZES` is sorted ascending; this `const fn` performs
/// ≤ log₂(64) = 6 comparisons. When `n` is a compile-time constant the
/// compiler resolves the entire search at compile time (zero runtime cost).
#[inline(always)]
const fn is_short_winograd_size(n: usize) -> bool {
    let mut lo = 0usize;
    let mut hi = SHORT_WINOGRAD_SIZES.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let mid_val = SHORT_WINOGRAD_SIZES[mid];
        if mid_val == n {
            return true;
        } else if mid_val < n {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    false
}

/// Macro-generated dispatch for every size in [`SHORT_WINOGRAD_SIZES`].
/// Adding or removing a codelet size must update both this invocation and
/// the `SHORT_WINOGRAD_SIZES` array.
macro_rules! short_winograd_match {
    ($data:ident, $F:ty, $INVERSE:ident, $($n:literal),+ $(,)?) => {
        match $data.len() {
            $(
                $n => {
                    let ptr = $data.as_mut_ptr() as *mut [num_complex::Complex<$F>; $n];
                    let arr = unsafe { &mut *ptr };
                    <$F as ShortDft<$n>>::dft::<$INVERSE>(arr);
                    true
                }
            )+
            _ => false,
        }
    };
}

#[inline(always)]
pub(crate) fn short_winograd<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>> + ShortWinogradScalar,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [num_complex::Complex<F>],
) -> bool {
    // Fast-reject: O(log N) binary search on the sorted const array.
    if !is_short_winograd_size(data.len()) {
        return false;
    }
    let handled = short_winograd_match!(
        data, F, INVERSE, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 60, 62, 63, 64, 81, 128
    );

    if handled && INVERSE && NORMALIZE {
        F::normalize(data, data.len());
    }

    handled
}

pub trait ShortDft<const N: usize>: ShortWinogradScalar {
    fn dft<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; N]);
}

macro_rules! impl_short_dft {
    ($n:expr, radix_dft2) => {
        impl<F: ShortWinogradScalar> ShortDft<$n> for F {
            #[inline(always)]
            fn dft<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; $n]) {
                Self::dft2(data);
            }
        }
    };
    ($n:expr, $method:ident) => {
        impl<F: ShortWinogradScalar> ShortDft<$n> for F {
            #[inline(always)]
            fn dft<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; $n]) {
                Self::$method::<INVERSE>(data);
            }
        }
    };
    ($n:expr, slice_method, $method:ident) => {
        impl<F: ShortWinogradScalar> ShortDft<$n> for F {
            #[inline(always)]
            fn dft<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; $n]) {
                Self::$method::<INVERSE>(data.as_mut_slice());
            }
        }
    };
    ($n:expr, winograd_impl, $func:ident) => {
        impl<F: ShortWinogradScalar> ShortDft<$n> for F {
            #[inline(always)]
            #[allow(unused_unsafe)]
            fn dft<const INVERSE: bool>(data: &mut [num_complex::Complex<Self>; $n]) {
                unsafe {
                    winograd::$func::<Self, INVERSE>(data);
                }
            }
        }
    };
}

impl_short_dft!(2, radix_dft2);
impl_short_dft!(3, dft3);
impl_short_dft!(4, dft4);
impl_short_dft!(5, dft5);
impl_short_dft!(6, winograd_impl, dft6_impl);
impl_short_dft!(7, dft7);
impl_short_dft!(8, dft8);
impl_short_dft!(9, winograd_impl, dft9_impl);
impl_short_dft!(10, winograd_impl, dft10_impl);
impl_short_dft!(11, dft11);
impl_short_dft!(12, winograd_impl, dft12_impl);
impl_short_dft!(13, dft13);
impl_short_dft!(14, winograd_impl, dft14_impl);
impl_short_dft!(15, winograd_impl, dft15_impl);
impl_short_dft!(16, dft16);
impl_short_dft!(17, dft17);
impl_short_dft!(18, winograd_impl, dft18_impl);
impl_short_dft!(19, dft19);
impl_short_dft!(20, winograd_impl, dft20_impl);
impl_short_dft!(21, winograd_impl, dft21_impl);
impl_short_dft!(22, winograd_impl, dft22_impl);
impl_short_dft!(23, dft23);
impl_short_dft!(24, winograd_impl, dft24_impl);
impl_short_dft!(25, winograd_impl, dft25_impl);
impl_short_dft!(26, winograd_impl, dft26_impl);
impl_short_dft!(27, winograd_impl, dft27_impl);
impl_short_dft!(28, winograd_impl, dft28_impl);
impl_short_dft!(29, dft29);
impl_short_dft!(30, winograd_impl, dft30_impl);
impl_short_dft!(31, dft31);
impl_short_dft!(32, winograd_impl, dft32_impl);
impl_short_dft!(33, winograd_impl, dft33_impl);
impl_short_dft!(34, winograd_impl, dft34_impl);
impl_short_dft!(35, winograd_impl, dft35_impl);
impl_short_dft!(36, winograd_impl, dft36_impl);
impl_short_dft!(37, dft37);
impl_short_dft!(38, winograd_impl, dft38_impl);
impl_short_dft!(39, winograd_impl, dft39_impl);
impl_short_dft!(40, winograd_impl, dft40_impl);
impl_short_dft!(41, dft41);
impl_short_dft!(42, winograd_impl, dft42_impl);
impl_short_dft!(43, dft43);
impl_short_dft!(44, winograd_impl, dft44_impl);
impl_short_dft!(45, winograd_impl, dft45_impl);
impl_short_dft!(46, winograd_impl, dft46_impl);
impl_short_dft!(47, dft47);
impl_short_dft!(48, winograd_impl, dft48_impl);
impl_short_dft!(49, winograd_impl, dft49_impl);
impl_short_dft!(50, winograd_impl, dft50_impl);
impl_short_dft!(51, winograd_impl, dft51_impl);
impl_short_dft!(52, winograd_impl, dft52_impl);
impl_short_dft!(53, dft53);
impl_short_dft!(54, winograd_impl, dft54_impl);
impl_short_dft!(55, winograd_impl, dft55_impl);
impl_short_dft!(56, winograd_impl, dft56_impl);
impl_short_dft!(58, winograd_impl, dft58_impl);
impl_short_dft!(60, winograd_impl, dft60_impl);
impl_short_dft!(62, winograd_impl, dft62_impl);
impl_short_dft!(63, winograd_impl, dft63_impl);
impl_short_dft!(64, winograd_impl, dft64_impl);
impl_short_dft!(81, winograd_impl, dft81_impl);
impl_short_dft!(128, winograd_impl, dft128_impl);

#[cfg(test)]
mod tests {
    use super::{short_winograd, SHORT_WINOGRAD_SIZES};
    use num_complex::Complex64;

    fn make_data(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| Complex64::new((k as f64 * 0.19).sin(), (k as f64 * 0.31).cos()))
            .collect()
    }

    #[test]
    fn short_winograd_fast_reject_sizes_not_in_array() {
        // Sizes absent from SHORT_WINOGRAD_SIZES must be rejected.
        let test_sizes = [0usize, 1, 57, 59, 61, 65, 100, 256];
        for &n in &test_sizes {
            assert!(
                !SHORT_WINOGRAD_SIZES.contains(&n),
                "size {n} must NOT be in SHORT_WINOGRAD_SIZES (test assumption)"
            );
            let mut data = make_data(n);
            let result = short_winograd::<f64, false, false>(&mut data);
            assert!(
                !result,
                "short_winograd must reject n={n} (not in SHORT_WINOGRAD_SIZES)"
            );
        }
    }

    #[test]
    fn short_winograd_accepts_sizes_in_array() {
        // Every size in SHORT_WINOGRAD_SIZES must be handled by the match.
        for &n in SHORT_WINOGRAD_SIZES {
            let mut data = make_data(n);
            let result = short_winograd::<f64, false, false>(&mut data);
            assert!(
                result,
                "short_winograd must handle n={n} (in SHORT_WINOGRAD_SIZES) but returned false"
            );
        }
    }
}
