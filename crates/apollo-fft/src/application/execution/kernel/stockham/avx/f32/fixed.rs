use super::triple_1::stage_triple32_radix1_avx_fma;
use super::triple_2::stage_triple32_quarter_groups_one_avx_fma;
use num_complex::Complex32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn fixed_len64_32_avx_fma(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    debug_assert_eq!(data.len(), 64);
    debug_assert!(scratch.len() >= 64);
    debug_assert!(twiddles.len() >= 63);

    let first_second = twiddles.get_unchecked(1..3);
    let first_third = twiddles.get_unchecked(3..7);
    let second_first = twiddles.get_unchecked(7..15);
    let second_second = twiddles.get_unchecked(15..31);
    let second_third = twiddles.get_unchecked(31..63);

    stage_triple32_radix1_avx_fma(data, scratch, first_second, first_third);
    stage_triple32_quarter_groups_one_avx_fma(
        scratch,
        data,
        8,
        second_first,
        second_second,
        second_third,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn cmul_vec32(
    w_re: std::arch::x86_64::__m256,
    w_im: std::arch::x86_64::__m256,
    value: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_mm256_fmaddsub_ps, _mm256_mul_ps, _mm256_permute_ps};
    _mm256_fmaddsub_ps(
        w_re,
        value,
        _mm256_mul_ps(w_im, _mm256_permute_ps::<0b1011_0001>(value)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn cmul_pair32(
    w_re: std::arch::x86_64::__m128,
    w_im: std::arch::x86_64::__m128,
    value: std::arch::x86_64::__m128,
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::{_mm_fmaddsub_ps, _mm_mul_ps, _mm_permute_ps};
    _mm_fmaddsub_ps(
        w_re,
        value,
        _mm_mul_ps(w_im, _mm_permute_ps::<0b1011_0001>(value)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn store_complex32_low(dst: *mut Complex32, value: std::arch::x86_64::__m128) {
    use std::arch::x86_64::{_mm_castps_si128, _mm_storel_epi64};
    _mm_storel_epi64(dst.cast(), _mm_castps_si128(value));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn store_complex32_high(dst: *mut Complex32, value: std::arch::x86_64::__m128) {
    use std::arch::x86_64::_mm_movehl_ps;
    store_complex32_low(dst, _mm_movehl_ps(value, value));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn avx_rotate_quarter_turn32(
    value: std::arch::x86_64::__m256,
    sign_mask: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_mm256_permute_ps, _mm256_xor_ps};

    _mm256_xor_ps(_mm256_permute_ps::<0b1011_0001>(value), sign_mask)
}
