use eunomia::Complex32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(crate) unsafe fn cmul_vec_reduced(
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
pub(crate) unsafe fn cmul_pair_reduced(
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
pub(crate) unsafe fn store_reduced_low(dst: *mut Complex32, value: std::arch::x86_64::__m128) {
    use std::arch::x86_64::{_mm_castps_si128, _mm_storel_epi64};
    _mm_storel_epi64(dst.cast(), _mm_castps_si128(value));
}
