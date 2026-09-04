//! Register-pressure-aware length-32 precise codelet.

use eunomia::Complex64;

#[cfg(target_arch = "x86_64")]
use super::super::simd::avx::{
    avx_cmul_precise, avx_fft4_parallel_precise, avx_fft8_parallel_precise, avx_fma_available,
};
#[cfg(target_arch = "x86_64")]
use super::super::twiddle_constants::{TWIDDLES_COMBINE_FWD_32, TWIDDLES_COMBINE_INV_32};

/// Runs the AVX/FMA length-32 codelet when the host supports its instructions.
pub(super) fn try_inplace<const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex64],
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if avx_fma_available() {
            // SAFETY: the capability probe establishes AVX and FMA support, and the
            // length-32 caller supplies the valid logical span required below.
            unsafe { vector_arm::<INVERSE, NORMALIZE>(data) };
            return true;
        }
    }

    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn stage_group<const GROUP: usize, const INVERSE: bool>(
    ptr: *mut f64,
    tw_ptr: *const f64,
) -> [std::arch::x86_64::__m256d; 4] {
    let [c0, mut c1, mut c2, mut c3] = unsafe {
        avx_fft4_parallel_precise::<INVERSE>(
            std::arch::x86_64::_mm256_loadu_pd(ptr.add(GROUP * 4)),
            std::arch::x86_64::_mm256_loadu_pd(ptr.add(16 + GROUP * 4)),
            std::arch::x86_64::_mm256_loadu_pd(ptr.add(32 + GROUP * 4)),
            std::arch::x86_64::_mm256_loadu_pd(ptr.add(48 + GROUP * 4)),
        )
    };

    // SAFETY: the twiddle table contains the four lanes for each of the three
    // nonzero radix-4 outputs at these compile-time offsets.
    unsafe {
        c1 = avx_cmul_precise(
            c1,
            std::arch::x86_64::_mm256_loadu_pd(tw_ptr.add(16 + GROUP * 4)),
        );
        c2 = avx_cmul_precise(
            c2,
            std::arch::x86_64::_mm256_loadu_pd(tw_ptr.add(32 + GROUP * 4)),
        );
        c3 = avx_cmul_precise(
            c3,
            std::arch::x86_64::_mm256_loadu_pd(tw_ptr.add(48 + GROUP * 4)),
        );
    }

    [c0, c1, c2, c3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn transpose_pair(
    left: std::arch::x86_64::__m256d,
    right: std::arch::x86_64::__m256d,
) -> [std::arch::x86_64::__m256d; 2] {
    [
        std::arch::x86_64::_mm256_permute2f128_pd::<0x20>(left, right),
        std::arch::x86_64::_mm256_permute2f128_pd::<0x31>(left, right),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn store_output<const INVERSE: bool, const NORMALIZE: bool>(
    ptr: *mut f64,
    output0: [std::arch::x86_64::__m256d; 8],
    output1: [std::arch::x86_64::__m256d; 8],
) {
    let scale = std::arch::x86_64::_mm256_set1_pd(1.0 / 32.0);
    for (row, (mut left, mut right)) in output0.into_iter().zip(output1).enumerate() {
        if INVERSE && NORMALIZE {
            left = std::arch::x86_64::_mm256_mul_pd(left, scale);
            right = std::arch::x86_64::_mm256_mul_pd(right, scale);
        }

        // SAFETY: the eight compile-time output rows and their two adjacent
        // vectors cover exactly the 64-double caller span.
        unsafe {
            let base = ptr.add(row * 8);
            std::arch::x86_64::_mm256_storeu_pd(base, left);
            std::arch::x86_64::_mm256_storeu_pd(base.add(4), right);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn vector_arm<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex64]) {
    let ptr = data.as_mut_ptr().cast::<f64>();
    let tw_table = if INVERSE {
        &TWIDDLES_COMBINE_INV_32
    } else {
        &TWIDDLES_COMBINE_FWD_32
    };
    let tw_ptr = tw_table.as_ptr().cast::<f64>();

    // All source vectors are loaded before the final stores, so the complete
    // transform remains in-place without a scratch allocation or a protected
    // load frontier.
    let mid0 = stage_group::<0, INVERSE>(ptr, tw_ptr);
    let mid1 = stage_group::<1, INVERSE>(ptr, tw_ptr);
    let mid2 = stage_group::<2, INVERSE>(ptr, tw_ptr);
    let mid3 = stage_group::<3, INVERSE>(ptr, tw_ptr);

    let [mid00, mid01] = transpose_pair(mid0[0], mid0[1]);
    let [mid10, mid11] = transpose_pair(mid1[0], mid1[1]);
    let [mid20, mid21] = transpose_pair(mid2[0], mid2[1]);
    let [mid30, mid31] = transpose_pair(mid3[0], mid3[1]);
    let output0 = unsafe {
        avx_fft8_parallel_precise::<INVERSE>(mid00, mid01, mid10, mid11, mid20, mid21, mid30, mid31)
    };

    let [mid02, mid03] = transpose_pair(mid0[2], mid0[3]);
    let [mid12, mid13] = transpose_pair(mid1[2], mid1[3]);
    let [mid22, mid23] = transpose_pair(mid2[2], mid2[3]);
    let [mid32, mid33] = transpose_pair(mid3[2], mid3[3]);
    let output1 = unsafe {
        avx_fft8_parallel_precise::<INVERSE>(mid02, mid03, mid12, mid13, mid22, mid23, mid32, mid33)
    };
    unsafe { store_output::<INVERSE, NORMALIZE>(ptr, output0, output1) };
}

#[cfg(test)]
mod tests {
    use super::Complex64;

    fn assert_matches_reference<const INVERSE: bool, const NORMALIZE: bool>() {
        let input: [Complex64; 32] = core::array::from_fn(|index| {
            let value = index as f64;
            Complex64::new((value * 0.17).sin(), (value * 0.29).cos())
        });
        let mut got = input;
        let mut expected = input;

        if !super::try_inplace::<INVERSE, NORMALIZE>(&mut got) {
            return;
        }
        crate::application::execution::kernel::components::winograd::dft32_impl::<f64, INVERSE>(
            &mut expected,
        );
        if INVERSE && NORMALIZE {
            let scale = Complex64::new(1.0 / 32.0, 0.0);
            for value in &mut expected {
                *value *= scale;
            }
        }

        let error = got
            .iter()
            .zip(expected.iter())
            .map(|(actual, reference)| (*actual - *reference).norm())
            .fold(0.0, f64::max);
        assert!(error < 1.0e-12, "n=32 f64 codelet error={error:e}");
    }

    #[test]
    fn forward_matches_reference() {
        assert_matches_reference::<false, false>();
    }

    #[test]
    fn forward_impulse_matches_reference() {
        let mut data = [Complex64::default(); 32];
        data[1] = Complex64::new(1.0, 0.0);
        let mut expected = data;
        crate::application::execution::kernel::components::winograd::dft32_impl::<f64, false>(
            &mut expected,
        );
        assert!(super::try_inplace::<false, false>(&mut data));
        for (index, value) in data.into_iter().enumerate() {
            assert!(
                (value - expected[index]).norm() < 1.0e-12,
                "impulse output index={index} value=({}, {})",
                value.re,
                value.im
            );
        }
    }

    #[test]
    fn inverse_matches_reference() {
        assert_matches_reference::<true, false>();
    }

    #[test]
    fn normalized_inverse_matches_reference() {
        assert_matches_reference::<true, true>();
    }
}
