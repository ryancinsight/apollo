use super::super::backend::StockhamAvxBackend;
use eunomia::Complex32;
use std::arch::x86_64::__m256;

impl StockhamAvxBackend for f32 {
    type Real = f32;
    type Complex = Complex32;
    type Vector = __m256;

    #[inline]
    unsafe fn mul(a: __m256, b: __m256) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_mul_ps(a, b) }
    }

    #[inline]
    unsafe fn fmaddsub(a: __m256, b: __m256, c: __m256) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_fmaddsub_ps(a, b, c) }
    }

    #[inline]
    unsafe fn permute_complex_swap(a: __m256) -> __m256 {
        unsafe { std::arch::x86_64::_mm256_permute_ps::<0b1011_0001>(a) }
    }

    #[inline]
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

    #[inline]
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

    #[inline]
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

    #[inline]
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
