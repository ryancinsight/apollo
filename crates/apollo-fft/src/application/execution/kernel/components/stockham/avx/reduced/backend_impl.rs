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

    #[inline(always)]
    unsafe fn rotate_quarter_turn(v: __m256, sign: f32) -> __m256 {
        let mask = unsafe {
            if sign > 0.0 {
                std::arch::x86_64::_mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0)
            } else {
                std::arch::x86_64::_mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0)
            }
        };
        unsafe { super::fixed::avx_rotate_quarter_turn_reduced(v, mask) }
    }

    #[inline(always)]
    unsafe fn stage_groups_one(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        twiddles: &[Complex32],
    ) {
        unsafe { super::base::stage_reduced_groups_one_avx_fma(src, dst, radix, twiddles) }
    }

    #[inline(always)]
    unsafe fn stage_pair_groups_two(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
    ) {
        unsafe {
            super::pair::stage_pair_groups_two_reduced_avx_fma(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
            )
        }
    }

    #[inline(always)]
    unsafe fn stage_pair_quarter_groups_two(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
    ) {
        unsafe {
            super::pair::stage_pair_quarter_groups_two_reduced_avx_fma(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
            )
        }
    }

    #[inline(always)]
    unsafe fn stage_triple_quarter_groups_one(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
    ) {
        unsafe {
            super::triple_2::stage_triple_quarter_groups_one_reduced_avx_fma(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
            )
        }
    }

    #[inline(always)]
    unsafe fn stage_triple_quarter_groups_two(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
    ) {
        unsafe {
            super::triple_2::stage_triple_quarter_groups_two_reduced_avx_fma(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
            )
        }
    }

    #[inline(always)]
    unsafe fn stockham_quad_groups_eight(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
        fourth_twiddles: &[Complex32],
    ) {
        unsafe {
            super::quad::stockham_quad_groups_eight_reduced(
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

    #[inline(always)]
    unsafe fn stockham_quad_groups_eight_low_live(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
        fourth_twiddles: &[Complex32],
    ) {
        unsafe {
            super::quad::stockham_quad_groups_eight_reduced(
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
