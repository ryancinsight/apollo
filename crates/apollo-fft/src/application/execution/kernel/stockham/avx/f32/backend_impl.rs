use super::super::backend::StockhamAvxBackend;
use num_complex::Complex32;
use std::arch::x86_64::__m256;

impl StockhamAvxBackend for f32 {
    type Real = f32;
    type Complex = Complex32;
    type Vector = __m256;

    const COMPLEX_PER_VECTOR: usize = 4;

    #[inline(always)]
    unsafe fn unpack_complex(c: Complex32) -> (f32, f32) {
        (c.re, c.im)
    }

    #[inline(always)]
    unsafe fn complex_mul(a: Complex32, b: Complex32) -> Complex32 {
        a * b
    }

    #[inline(always)]
    unsafe fn complex_add(a: Complex32, b: Complex32) -> Complex32 {
        a + b
    }

    #[inline(always)]
    unsafe fn complex_sub(a: Complex32, b: Complex32) -> Complex32 {
        a - b
    }

    #[inline(always)]
    unsafe fn set1_real(val: f32) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_set1_ps(val) }
    }

    #[inline(always)]
    unsafe fn set1_imag(val: f32) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_set1_ps(val) }
    }

    #[inline(always)]
    unsafe fn loadu_complex(ptr: *const Complex32) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_loadu_ps(ptr.cast::<f32>()) }
    }

    #[inline(always)]
    unsafe fn storeu_complex(ptr: *mut Complex32, val: __m256) {
        unsafe { std::arch::x86_64::_mm256_storeu_ps(ptr.cast::<f32>(), val) }
    }

    #[inline(always)]
    unsafe fn add(a: __m256, b: __m256) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_add_ps(a, b) }
    }

    #[inline(always)]
    unsafe fn sub(a: __m256, b: __m256) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_sub_ps(a, b) }
    }

    #[inline(always)]
    unsafe fn mul(a: __m256, b: __m256) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_mul_ps(a, b) }
    }

    #[inline(always)]
    unsafe fn fmaddsub(a: __m256, b: __m256, c: __m256) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_fmaddsub_ps(a, b, c) }
    }

    #[inline(always)]
    unsafe fn permute_complex_swap(a: __m256) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_permute_ps::<0b1011_0001>(a) }
    }
}
