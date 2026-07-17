//! `ShortWinogradScalar` trait — short-DFT codelet dispatch for small primes.
//!
//! This trait extends `WinogradScalar` with per-size DFT methods for canonical
//! Winograd short-DFT sizes (2..=53, plus composite codelets up to 484).
//!
//! Canonical definition lives here in the `components::winograd` hierarchy
//! next to its parent trait `WinogradScalar`, following deep-vertical SSOT.
//!
use super::radix::odd_prime_pair::{dft_pair_impl, dft_pair_impl_reduced, PrimePairTable};
use super::WinogradScalar;
use eunomia::{Complex32, Complex64};

macro_rules! impl_short_winograd_prime_pair {
    ($ty:ty, $(($method:ident, $n:expr, $h:expr)),+ $(,)?) => {
        $(
            #[inline]
            fn $method<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; $n]) {
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
            #[inline]
            fn $method<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; $n]) {
                dft_pair_impl_reduced::<f32, $n, $h, INVERSE>(
                    data,
                    <f32 as PrimePairTable<$n, $h>>::cos_table(),
                    <f32 as PrimePairTable<$n, $h>>::sin_table(),
                );
            }
        )+
    };
}

pub trait ShortWinogradScalar: WinogradScalar {
    fn dft2(data: &mut [eunomia::Complex<Self>; 2]);
    fn dft3<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 3]);
    fn dft4<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 4]);
    fn dft5<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 5]);
    fn dft7<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 7]);
    fn dft8<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 8]);
    fn dft16<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 16]);
    fn dft11<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 11]);
    fn dft13<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 13]);
    fn dft17<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 17]);
    fn dft19<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 19]);
    fn dft23<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 23]);
    fn dft29<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 29]);
    fn dft31<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 31]);
    fn dft37<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 37]);
    fn dft41<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 41]);
    fn dft43<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 43]);
    fn dft47<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 47]);
    fn dft53<const INVERSE: bool>(data: &mut [eunomia::Complex<Self>; 53]);
}

impl ShortWinogradScalar for f64 {
    #[inline]
    fn dft2(data: &mut [Complex64; 2]) {
        super::dft2_impl(data);
    }

    #[inline]
    fn dft3<const INVERSE: bool>(data: &mut [Complex64; 3]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        {
            use std::arch::x86_64::{
                _mm_add_pd, _mm_fmadd_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_set_pd,
                _mm_shuffle_pd, _mm_storeu_pd, _mm_sub_pd,
            };
            // SAFETY: `data` is a contiguous three-element `Complex64` array, so
            // the six scalar lanes loaded and stored through `ptr` are in-bounds.
            unsafe {
                let ptr = data.as_mut_ptr().cast::<f64>();
                let x0 = _mm_loadu_pd(ptr);
                let x1 = _mm_loadu_pd(ptr.add(2));
                let x2 = _mm_loadu_pd(ptr.add(4));
                let sum = _mm_add_pd(x1, x2);
                let diff = _mm_sub_pd(x1, x2);
                let m0 = _mm_fmadd_pd(sum, _mm_set1_pd(-0.5), x0);
                let diff_swapped = _mm_shuffle_pd(diff, diff, 0x1);
                let sign = if INVERSE {
                    _mm_set_pd(1.0, -1.0)
                } else {
                    _mm_set_pd(-1.0, 1.0)
                };
                let m1 = _mm_mul_pd(
                    _mm_mul_pd(diff_swapped, sign),
                    _mm_set1_pd(0.8660254037844386),
                );
                _mm_storeu_pd(ptr, _mm_add_pd(x0, sum));
                _mm_storeu_pd(ptr.add(2), _mm_add_pd(m0, m1));
                _mm_storeu_pd(ptr.add(4), _mm_sub_pd(m0, m1));
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
        #[allow(unreachable_code)]
        {
            super::dft3_impl::<f64, INVERSE, false>(data);
        }
    }

    #[inline]
    fn dft4<const INVERSE: bool>(data: &mut [Complex64; 4]) {
        super::dft4_array_impl::<f64, INVERSE, false>(data);
    }

    #[inline]
    fn dft5<const INVERSE: bool>(data: &mut [Complex64; 5]) {
        super::dft5_array_impl::<f64, INVERSE, false>(data);
    }

    #[inline]
    fn dft7<const INVERSE: bool>(data: &mut [Complex64; 7]) {
        super::dft7_impl::<f64, INVERSE, false>(data);
    }

    #[inline]
    fn dft8<const INVERSE: bool>(data: &mut [Complex64; 8]) {
        super::dft8_array_impl::<f64, INVERSE, false>(data);
    }

    #[inline]
    fn dft16<const INVERSE: bool>(data: &mut [Complex64; 16]) {
        super::dft16_impl::<f64, INVERSE>(data);
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
    #[inline]
    fn dft2(data: &mut [Complex32; 2]) {
        super::dft2_impl(data);
    }

    #[inline]
    fn dft3<const INVERSE: bool>(data: &mut [Complex32; 3]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        {
            use std::arch::x86_64::{
                _mm_add_ps, _mm_fmadd_ps, _mm_loadu_ps, _mm_movelh_ps, _mm_mul_ps, _mm_set1_ps,
                _mm_set_ps, _mm_shuffle_ps, _mm_store_ss, _mm_storeu_ps, _mm_sub_ps,
            };
            // SAFETY: `data` is a contiguous three-element `Complex32` array, so
            // the six scalar lanes loaded and stored through `ptr` are in-bounds.
            unsafe {
                let ptr = data.as_mut_ptr().cast::<f32>();
                let v0 = _mm_loadu_ps(ptr);
                let v12 = _mm_loadu_ps(ptr.add(2));
                let x0 = _mm_shuffle_ps::<0x44>(v0, v0);
                let x1 = _mm_shuffle_ps::<0xEE>(v0, v0);
                let x2 = _mm_shuffle_ps::<0xEE>(v12, v12);
                let sum = _mm_add_ps(x1, x2);
                let diff = _mm_sub_ps(x1, x2);
                let m0 = _mm_fmadd_ps(sum, _mm_set1_ps(-0.5), x0);
                let diff_swapped = _mm_shuffle_ps::<0xB1>(diff, diff);
                let sign = if INVERSE {
                    _mm_set_ps(1.0, -1.0, 1.0, -1.0)
                } else {
                    _mm_set_ps(-1.0, 1.0, -1.0, 1.0)
                };
                let m1 = _mm_mul_ps(
                    _mm_mul_ps(diff_swapped, sign),
                    _mm_set1_ps(0.8660254037844386),
                );
                let out01 = _mm_movelh_ps(_mm_add_ps(x0, sum), _mm_add_ps(m0, m1));
                _mm_storeu_ps(ptr, out01);
                let out2 = _mm_sub_ps(m0, m1);
                _mm_store_ss(ptr.add(4), out2);
                _mm_store_ss(ptr.add(5), _mm_shuffle_ps::<0xE5>(out2, out2));
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
        #[allow(unreachable_code)]
        {
            super::dft3_impl::<f32, INVERSE, false>(data);
        }
    }

    #[inline]
    fn dft4<const INVERSE: bool>(data: &mut [Complex32; 4]) {
        super::dft4_array_impl::<f32, INVERSE, false>(data);
    }

    #[inline]
    fn dft5<const INVERSE: bool>(data: &mut [Complex32; 5]) {
        super::dft5_array_impl::<f32, INVERSE, false>(data);
    }

    #[inline]
    fn dft7<const INVERSE: bool>(data: &mut [Complex32; 7]) {
        super::dft7_impl::<f32, INVERSE, false>(data);
    }

    #[inline]
    fn dft8<const INVERSE: bool>(data: &mut [Complex32; 8]) {
        super::dft8_array_impl::<f32, INVERSE, false>(data);
    }

    #[inline]
    fn dft16<const INVERSE: bool>(data: &mut [Complex32; 16]) {
        super::dft16_impl::<f32, INVERSE>(data);
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
