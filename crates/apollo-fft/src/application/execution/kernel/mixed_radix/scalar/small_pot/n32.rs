//! Register-pressure-aware length-32 precise codelet.

#[cfg(target_arch = "x86_64")]
use core::mem::MaybeUninit;

use eunomia::Complex64;

#[cfg(target_arch = "x86_64")]
use super::super::simd::avx::{
    avx_cmul_precise, avx_fft4_parallel_precise, avx_fft8_parallel_precise, avx_fma_available,
};
#[cfg(target_arch = "x86_64")]
use super::super::twiddle_constants::{TWIDDLES_COMBINE_FWD_32, TWIDDLES_COMBINE_INV_32};

/// Runs the AVX/FMA codelet over exactly 32 complex values when supported.
///
/// The array borrow establishes the complete span required by the vector loads
/// and stores. An unsupported host leaves the array unchanged.
pub(super) fn try_inplace<const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex64; 32],
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if avx_fma_available() {
            // SAFETY: the probe establishes AVX and FMA support; the exclusive
            // array borrow establishes 32 initialized, writable complex values.
            unsafe { vector_arm::<INVERSE, NORMALIZE>(data) };
            return true;
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    let _ = data;

    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn stage_group<const GROUP: usize, const INVERSE: bool>(
    ptr: *mut f64,
    tw_ptr: *const f64,
) -> [std::arch::x86_64::__m256d; 4] {
    // SAFETY: GROUP is 0..4 at every instantiation. Each load reads four
    // doubles within the caller's 64-double array, with AVX/FMA enabled.
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
unsafe fn vector_arm<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex64; 32]) {
    let ptr = data.as_mut_ptr().cast::<f64>();
    let tw_table = if INVERSE {
        &TWIDDLES_COMBINE_INV_32
    } else {
        &TWIDDLES_COMBINE_FWD_32
    };
    let tw_ptr = tw_table.as_ptr().cast::<f64>();

    // Both final halves read the intermediate, so stores to data cannot
    // overwrite inputs still needed by the other half.
    let mut middle = MaybeUninit::<[f64; 64]>::uninit();
    let middle_ptr = middle.as_mut_ptr().cast::<f64>();
    // SAFETY: the four groups read within data and initialize disjoint,
    // exhaustive 16-double regions of middle. The selected twiddle table has
    // all 32 complex coefficients. AVX/FMA is enabled for this function.
    unsafe {
        store_stage_group::<0>(middle_ptr, stage_group::<0, INVERSE>(ptr, tw_ptr));
        store_stage_group::<1>(middle_ptr, stage_group::<1, INVERSE>(ptr, tw_ptr));
        store_stage_group::<2>(middle_ptr, stage_group::<2, INVERSE>(ptr, tw_ptr));
        store_stage_group::<3>(middle_ptr, stage_group::<3, INVERSE>(ptr, tw_ptr));
    }

    // SAFETY: all four stage groups initialized the complete intermediate.
    let output0 = unsafe {
        let [mid00, mid01] = load_stage_half::<0, 0>(middle_ptr);
        let [mid10, mid11] = load_stage_half::<1, 0>(middle_ptr);
        let [mid20, mid21] = load_stage_half::<2, 0>(middle_ptr);
        let [mid30, mid31] = load_stage_half::<3, 0>(middle_ptr);
        avx_fft8_parallel_precise::<INVERSE>(mid00, mid01, mid10, mid11, mid20, mid21, mid30, mid31)
    };
    // SAFETY: OFFSET = 0 covers the first adjacent vector of each output row.
    unsafe { store_output_half::<INVERSE, NORMALIZE, 0>(ptr, output0) };

    // SAFETY: the second half reads initialized intermediate lanes, not data.
    let output1 = unsafe {
        let [mid02, mid03] = load_stage_half::<0, 1>(middle_ptr);
        let [mid12, mid13] = load_stage_half::<1, 1>(middle_ptr);
        let [mid22, mid23] = load_stage_half::<2, 1>(middle_ptr);
        let [mid32, mid33] = load_stage_half::<3, 1>(middle_ptr);
        avx_fft8_parallel_precise::<INVERSE>(mid02, mid03, mid12, mid13, mid22, mid23, mid32, mid33)
    };
    // SAFETY: OFFSET = 4 covers the second adjacent vector of each output row.
    unsafe { store_output_half::<INVERSE, NORMALIZE, 4>(ptr, output1) };
}

#[cfg(test)]
mod tests {
    use super::Complex64;

    fn signal() -> [Complex64; 32] {
        core::array::from_fn(|index| {
            let value = index as f64;
            Complex64::new((value * 0.17).sin(), (value * 0.29).cos())
        })
    }

    fn assert_matches_reference<const INVERSE: bool, const NORMALIZE: bool>(
        input: &[Complex64; 32],
    ) {
        #[cfg(target_arch = "x86_64")]
        let supported =
            std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let supported = false;

        let mut expected = *input;
        crate::application::execution::kernel::components::winograd::dft32_impl::<f64, INVERSE>(
            &mut expected,
        );
        if INVERSE && NORMALIZE {
            let scale = Complex64::new(1.0 / 32.0, 0.0);
            for value in &mut expected {
                *value *= scale;
            }
        }

        let sentinel = Complex64::new(123.0, -456.0);
        // Adjacent complex offsets exercise both 16-byte alignment residues
        // modulo the 32-byte AVX load width, with guards on both sides.
        for offset in [1, 2] {
            let mut guarded = [sentinel; 36];
            let (prefix, tail) = guarded.split_at_mut(offset);
            let (values, suffix) = tail.split_at_mut(32);
            let got: &mut [Complex64; 32] = values.try_into().expect("exact codelet span");
            got.copy_from_slice(input);

            let executed = super::try_inplace::<INVERSE, NORMALIZE>(got);
            assert_eq!(executed, supported, "dispatch must match host capabilities");
            if executed {
                for (index, (actual, reference)) in got.iter().zip(&expected).enumerate() {
                    let error = (*actual - *reference).norm();
                    // Elementwise comparison also rejects NaN; a max reduction
                    // can discard it and report a finite error for bad output.
                    assert!(
                        error < 1.0e-12,
                        "n=32 codelet index={index} error={error:e}"
                    );
                }
            } else {
                assert_eq!(got, input, "unsupported host must preserve the input");
            }
            assert_eq!(prefix, &[sentinel; 2][..offset]);
            assert_eq!(suffix, &[sentinel; 3][..4 - offset]);
        }
    }

    #[test]
    fn forward_matches_reference() {
        assert_matches_reference::<false, false>(&signal());
    }

    #[test]
    fn forward_normalization_flag_matches_reference() {
        assert_matches_reference::<false, true>(&signal());
    }

    #[test]
    fn forward_impulse_matches_reference() {
        let mut data = [Complex64::default(); 32];
        data[1] = Complex64::new(1.0, 0.0);
        assert_matches_reference::<false, false>(&data);
    }

    #[test]
    fn inverse_matches_reference() {
        assert_matches_reference::<true, false>(&signal());
    }

    #[test]
    fn normalized_inverse_matches_reference() {
        assert_matches_reference::<true, true>(&signal());
    }
}
