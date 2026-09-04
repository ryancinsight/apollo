//! Register-pressure-aware length-32 precise codelet.

use core::mem::MaybeUninit;

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
unsafe fn store_stage_group<const GROUP: usize>(
    ptr: *mut f64,
    values: [std::arch::x86_64::__m256d; 4],
) {
    for (index, value) in values.into_iter().enumerate() {
        // SAFETY: each stage group owns four adjacent vectors in the 64-double
        // intermediate, and the caller supplies the complete scratch span.
        unsafe { std::arch::x86_64::_mm256_storeu_pd(ptr.add(GROUP * 16 + index * 4), value) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn load_stage_half<const GROUP: usize, const HALF: usize>(
    ptr: *const f64,
) -> [std::arch::x86_64::__m256d; 2] {
    // SAFETY: stage_group wrote every vector before the final half is loaded;
    // each compile-time offset is within the 64-double intermediate.
    unsafe {
        transpose_pair(
            std::arch::x86_64::_mm256_loadu_pd(ptr.add(GROUP * 16 + HALF * 8)),
            std::arch::x86_64::_mm256_loadu_pd(ptr.add(GROUP * 16 + HALF * 8 + 4)),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn store_output_half<const INVERSE: bool, const NORMALIZE: bool, const OFFSET: usize>(
    ptr: *mut f64,
    output: [std::arch::x86_64::__m256d; 8],
) {
    let scale = std::arch::x86_64::_mm256_set1_pd(1.0 / 32.0);
    for (row, mut value) in output.into_iter().enumerate() {
        if INVERSE && NORMALIZE {
            value = std::arch::x86_64::_mm256_mul_pd(value, scale);
        }

        // SAFETY: the eight compile-time output rows and the selected adjacent
        // vector cover one half of the 64-double caller span.
        unsafe {
            std::arch::x86_64::_mm256_storeu_pd(ptr.add(row * 8 + OFFSET), value);
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

    // The sixteen radix-4 vectors cannot coexist with the sixteen radix-8
    // results in AVX2. This explicit stack intermediate makes the spill point
    // deterministic and releases each output half before the next half runs.
    let mut middle = MaybeUninit::<[f64; 64]>::uninit();
    let middle_ptr = middle.as_mut_ptr().cast::<f64>();
    unsafe {
        store_stage_group::<0>(middle_ptr, stage_group::<0, INVERSE>(ptr, tw_ptr));
        store_stage_group::<1>(middle_ptr, stage_group::<1, INVERSE>(ptr, tw_ptr));
        store_stage_group::<2>(middle_ptr, stage_group::<2, INVERSE>(ptr, tw_ptr));
        store_stage_group::<3>(middle_ptr, stage_group::<3, INVERSE>(ptr, tw_ptr));
    }

    let output0 = unsafe {
        let [mid00, mid01] = load_stage_half::<0, 0>(middle_ptr);
        let [mid10, mid11] = load_stage_half::<1, 0>(middle_ptr);
        let [mid20, mid21] = load_stage_half::<2, 0>(middle_ptr);
        let [mid30, mid31] = load_stage_half::<3, 0>(middle_ptr);
        avx_fft8_parallel_precise::<INVERSE>(mid00, mid01, mid10, mid11, mid20, mid21, mid30, mid31)
    };
    unsafe { store_output_half::<INVERSE, NORMALIZE, 0>(ptr, output0) };

    let output1 = unsafe {
        let [mid02, mid03] = load_stage_half::<0, 1>(middle_ptr);
        let [mid12, mid13] = load_stage_half::<1, 1>(middle_ptr);
        let [mid22, mid23] = load_stage_half::<2, 1>(middle_ptr);
        let [mid32, mid33] = load_stage_half::<3, 1>(middle_ptr);
        avx_fft8_parallel_precise::<INVERSE>(mid02, mid03, mid12, mid13, mid22, mid23, mid32, mid33)
    };
    unsafe { store_output_half::<INVERSE, NORMALIZE, 4>(ptr, output1) };
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
