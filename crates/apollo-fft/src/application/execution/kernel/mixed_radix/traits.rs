//! `ShortWinogradScalar` and generic short-DFT dispatch helpers.
//!
//! `ShortWinogradScalar` is canonically defined in
//! `components::winograd::short_winograd` (deep-vertical SSOT next to parent
//! `WinogradScalar`). This module re-exports it for backward compatibility.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

pub use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;

/// Canonical catalog of sizes with dedicated Winograd short-DFT codelets.
/// Each entry corresponds to a `ShortDft<N>` impl and a `short_winograd`
/// match arm. Adding or removing a codelet size must update this array.
/// The array **must remain sorted** (ascending) for `is_short_winograd_size`.
pub(crate) const SHORT_WINOGRAD_SIZES: &[usize] = &[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 58, 60, 62, 63, 64, 72, 81, 96, 99, 108, 112, 120, 121, 126, 128, 144, 154,
    168, 180, 189, 222, 242, 246, 259, 275, 280, 296, 363, 400, 484,
];

/// O(log N) membership test for `SHORT_WINOGRAD_SIZES` via binary search.
///
/// `SHORT_WINOGRAD_SIZES` is sorted ascending; this `const fn` performs
/// ≤ log₂(64) = 6 comparisons. When `n` is a compile-time constant the
/// compiler resolves the entire search at compile time (zero runtime cost).
#[inline]
pub(crate) const fn is_short_winograd_size(n: usize) -> bool {
    let mut left = 0;
    let mut right = SHORT_WINOGRAD_SIZES.len();
    while left < right {
        let mid = left + ((right - left) / 2);
        let candidate = SHORT_WINOGRAD_SIZES[mid];
        if candidate == n {
            return true;
        }
        if candidate < n {
            left = mid + 1;
        } else {
            right = mid;
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
                    let ptr = $data.as_mut_ptr() as *mut [eunomia::Complex<$F>; $n];
                    let arr = unsafe { &mut *ptr };
                    <$F as ShortDft<$n>>::dft::<$INVERSE>(arr);
                    true
                }
            )+
            _ => false,
        }
    };
}

#[inline]
pub(crate) fn short_winograd<
    F: MixedRadixScalar<Complex = eunomia::Complex<F>> + ShortWinogradScalar,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [eunomia::Complex<F>],
) -> bool {
    // Fast-reject: O(log N) binary search on the sorted const array.
    if !is_short_winograd_size(data.len()) {
        return false;
    }
    // For sizes >64, only use short Winograd codelet if the scalar explicitly selects it
    // via use_generated_codelet_plan (f32 list reduced to drop slow "Precision Policy" sizes).
    // This fixes suboptimal method selection for many worst benchmark cases; falls through
    // to composite / GT / Rader which may have better codegen/perf.
    if data.len() > 64 && !F::use_generated_codelet_plan(data.len()) {
        return false;
    }
    let handled = short_winograd_match!(
        data, F, INVERSE,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 60, 62, 63, 64, 72, 81, 96, 99, 108, 112,
        120, 121, 126, 128, 144, 154, 168, 180, 189, 222, 242, 246, 259, 275, 280, 296, 363, 400, 484
    );

    if handled && INVERSE && NORMALIZE {
        F::normalize(data, data.len());
    }

    handled
}

pub trait ShortDft<const N: usize>: ShortWinogradScalar {
    fn dft<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; N]);
}

macro_rules! impl_short_dft {
    ($n:expr, radix_dft2) => {
        impl<F: ShortWinogradScalar> ShortDft<$n> for F {
            #[inline]
            fn dft<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; $n]) {
                Self::dft2(data);
            }
        }
    };
    ($n:expr, $method:ident) => {
        impl<F: ShortWinogradScalar> ShortDft<$n> for F {
            #[inline]
            fn dft<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; $n]) {
                Self::$method::<INVERSE>(data);
            }
        }
    };
    ($n:expr, slice_method, $method:ident) => {
        impl<F: ShortWinogradScalar> ShortDft<$n> for F {
            #[inline]
            fn dft<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; $n]) {
                Self::$method::<INVERSE>(data.as_mut_slice());
            }
        }
    };
    ($n:expr, winograd_impl, $func:ident) => {
        impl<F: ShortWinogradScalar> ShortDft<$n> for F {
            #[inline]
            #[allow(unused_unsafe)]
            fn dft<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; $n]) {
                unsafe {
                    crate::application::execution::kernel::components::butterflies::dft::$func::<
                        Self,
                        INVERSE,
                    >(data);
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
impl_short_dft!(72, winograd_impl, dft72_impl);
impl_short_dft!(81, winograd_impl, dft81_impl);
impl_short_dft!(96, winograd_impl, dft96_impl);
impl_short_dft!(99, winograd_impl, dft99_impl);
impl_short_dft!(108, winograd_impl, dft108_impl);
impl_short_dft!(112, winograd_impl, dft112_impl);
impl_short_dft!(120, winograd_impl, dft120_impl);
impl_short_dft!(121, winograd_impl, dft121_impl);
impl_short_dft!(126, winograd_impl, dft126_impl);
impl_short_dft!(128, winograd_impl, dft128_impl);
impl_short_dft!(144, winograd_impl, dft144_impl);
impl_short_dft!(154, winograd_impl, dft154_impl);
impl_short_dft!(168, winograd_impl, dft168_impl);
impl_short_dft!(180, winograd_impl, dft180_impl);
impl_short_dft!(189, winograd_impl, dft189_impl);
impl_short_dft!(242, winograd_impl, dft242_impl);
impl_short_dft!(275, winograd_impl, dft275_impl);
impl_short_dft!(280, winograd_impl, dft280_impl);
impl_short_dft!(363, winograd_impl, dft363_impl);
impl_short_dft!(400, winograd_impl, dft400_impl);
impl_short_dft!(484, winograd_impl, dft484_impl);
impl_short_dft!(222, winograd_impl, dft222_impl);
impl_short_dft!(246, winograd_impl, dft246_impl);
impl_short_dft!(259, winograd_impl, dft259_impl);
impl_short_dft!(296, winograd_impl, dft296_impl);

#[cfg(test)]
mod tests {
    use super::{short_winograd, SHORT_WINOGRAD_SIZES};
    use eunomia::Complex64;

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
        // Every size in SHORT_WINOGRAD_SIZES must be handled by the match when the guard in
        // short_winograd allows (for f64 use_generated always false so >64 rejected here;
        // f32 policy may accept subset; list+match+impls keep support for when policy enables).
        for &n in SHORT_WINOGRAD_SIZES {
            if n > 64 {
                continue;
            }
            let mut data = make_data(n);
            let result = short_winograd::<f64, false, false>(&mut data);
            assert!(
                result,
                "short_winograd must handle n={n} (in SHORT_WINOGRAD_SIZES) but returned false"
            );
        }
    }
}
