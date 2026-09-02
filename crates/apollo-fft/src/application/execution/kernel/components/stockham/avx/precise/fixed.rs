//! Register-resident complex multiply shared by the AVX group specialisations.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn cmul_vec_precise(
    w_re: std::arch::x86_64::__m256d,
    w_im: std::arch::x86_64::__m256d,
    value: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{_mm256_fmaddsub_pd, _mm256_mul_pd, _mm256_permute_pd};
    _mm256_fmaddsub_pd(
        w_re,
        value,
        _mm256_mul_pd(w_im, _mm256_permute_pd::<0b0101>(value)),
    )
}
