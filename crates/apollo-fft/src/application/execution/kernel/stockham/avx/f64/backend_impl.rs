use super::super::backend::StockhamAvxBackend;
use num_complex::Complex64;
use std::arch::x86_64::__m256d;

impl StockhamAvxBackend for f64 {
    type Real = f64;
    type Complex = Complex64;
    type Vector = __m256d;

    const COMPLEX_PER_VECTOR: usize = 2;

    #[inline(always)]
    unsafe fn unpack_complex(c: Complex64) -> (f64, f64) {
        (c.re, c.im)
    }

    #[inline(always)]
    unsafe fn complex_mul(a: Complex64, b: Complex64) -> Complex64 {
        a * b
    }

    #[inline(always)]
    unsafe fn complex_add(a: Complex64, b: Complex64) -> Complex64 {
        a + b
    }

    #[inline(always)]
    unsafe fn complex_sub(a: Complex64, b: Complex64) -> Complex64 {
        a - b
    }

    #[inline(always)]
    unsafe fn set1_real(val: f64) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_set1_pd(val) }
    }

    #[inline(always)]
    unsafe fn set1_imag(val: f64) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_set1_pd(val) }
    }

    #[inline(always)]
    unsafe fn loadu_complex(ptr: *const Complex64) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_loadu_pd(ptr.cast::<f64>()) }
    }

    #[inline(always)]
    unsafe fn storeu_complex(ptr: *mut Complex64, val: __m256d) {
        unsafe { std::arch::x86_64::_mm256_storeu_pd(ptr.cast::<f64>(), val) }
    }

    #[inline(always)]
    unsafe fn add(a: __m256d, b: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_add_pd(a, b) }
    }

    #[inline(always)]
    unsafe fn sub(a: __m256d, b: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_sub_pd(a, b) }
    }

    #[inline(always)]
    unsafe fn mul(a: __m256d, b: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_mul_pd(a, b) }
    }

    #[inline(always)]
    unsafe fn fmaddsub(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_fmaddsub_pd(a, b, c) }
    }

    #[inline(always)]
    unsafe fn permute_complex_swap(a: __m256d) -> __m256d {
        unsafe { std::arch::x86_64::_mm256_permute_pd::<0b0101>(a) }
    }
}
