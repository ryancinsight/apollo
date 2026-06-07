use super::super::backend::StockhamAvxBackend;
use num_complex::Complex64;
use std::arch::x86_64::__m256d;

impl StockhamAvxBackend for f64 {
    type Real = f64;
    type Complex = Complex64;
    type Vector = __m256d;

    const COMPLEX_PER_VECTOR: usize = 2;

    #[inline]
    unsafe fn unpack_complex(c: Complex64) -> (f64, f64) {
        (c.re, c.im)
    }

    #[inline]
    unsafe fn complex_mul(a: Complex64, b: Complex64) -> Complex64 {
        a * b
    }

    #[inline]
    unsafe fn complex_add(a: Complex64, b: Complex64) -> Complex64 {
        a + b
    }

    #[inline]
    unsafe fn complex_sub(a: Complex64, b: Complex64) -> Complex64 {
        a - b
    }

    #[inline]
    unsafe fn set1_real(val: f64) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_set1_pd(val) }
    }

    #[inline]
    unsafe fn set1_imag(val: f64) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_set1_pd(val) }
    }

    #[inline]
    unsafe fn loadu_complex(ptr: *const Complex64) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_loadu_pd(ptr.cast::<f64>()) }
    }

    #[inline]
    unsafe fn storeu_complex(ptr: *mut Complex64, val: __m256d) {
        unsafe { std::arch::x86_64::_mm256_storeu_pd(ptr.cast::<f64>(), val) }
    }

    #[inline]
    unsafe fn add(a: __m256d, b: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_add_pd(a, b) }
    }

    #[inline]
    unsafe fn sub(a: __m256d, b: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_sub_pd(a, b) }
    }

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
    unsafe fn rotate_quarter_turn(v: __m256d, sign: f64) -> __m256d {
        let mask = unsafe {
            if sign > 0.0 {
                std::arch::x86_64::_mm256_set_pd(0.0, -0.0, 0.0, -0.0)
            } else {
                std::arch::x86_64::_mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
            }
        };
        unsafe { super::fixed::avx_rotate_quarter_turn(v, mask) }
    }

    #[inline]
    unsafe fn stage_groups_one(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        twiddles: &[Complex64],
    ) {
        unsafe { super::base::stage_precise_groups_one_avx_fma(src, dst, radix, twiddles) }
    }

    #[inline]
    unsafe fn stage_pair_groups_two(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
    ) {
        unsafe {
            super::pair::stage_pair_groups_two_precise_avx_fma(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
            )
        }
    }

    #[inline]
    unsafe fn stage_triple_quarter_groups_one(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) {
        unsafe {
            super::triple_1::stage_triple_quarter_groups_one_precise_avx_fma(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
            )
        }
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
