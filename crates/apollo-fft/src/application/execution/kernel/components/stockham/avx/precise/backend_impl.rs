use super::super::backend::StockhamAvxBackend;
use eunomia::Complex64;
use std::arch::x86_64::__m256d;

impl StockhamAvxBackend for f64 {
    type Real = f64;
    type Complex = Complex64;
    type Vector = __m256d;

    #[inline]
    unsafe fn mul(a: __m256d, b: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_mul_pd(a, b) }
    }

    #[inline]
    unsafe fn fmaddsub(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_fmaddsub_pd(a, b, c) }
    }

    #[inline]
    unsafe fn permute_complex_swap(a: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_permute_pd::<0b0101>(a) }
    }

    #[inline]
    unsafe fn stockham_quad_groups_eight_low_live(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
        fourth_twiddles: &[Complex64],
    ) {
        unsafe {
            super::quad::stockham_quad_groups_eight_precise(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
                fourth_twiddles,
            )
        }
    }
}
