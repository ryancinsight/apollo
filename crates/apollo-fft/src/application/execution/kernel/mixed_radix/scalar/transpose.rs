use num_complex::{Complex32, Complex64};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn transpose_avx_precise(src: &[Complex64], dst: &mut [Complex64], n1: usize, n2: usize) {
    use std::arch::x86_64::*;
    const TILE: usize = 16;
    for i_base in (0..n1).step_by(TILE) {
        for j_base in (0..n2).step_by(TILE) {
            let i_end = (i_base + TILE).min(n1);
            let j_end = (j_base + TILE).min(n2);
            let mut i = i_base;
            while i + 2 <= i_end {
                let mut j = j_base;
                while j + 2 <= j_end {
                    let r0 = _mm256_loadu_pd(src.as_ptr().add(i * n2 + j) as *const f64);
                    let r1 = _mm256_loadu_pd(src.as_ptr().add((i + 1) * n2 + j) as *const f64);
                    let top = _mm256_permute2f128_pd(r0, r1, 0x20);
                    let bot = _mm256_permute2f128_pd(r0, r1, 0x31);
                    _mm256_storeu_pd(dst.as_mut_ptr().add(j * n1 + i) as *mut f64, top);
                    _mm256_storeu_pd(dst.as_mut_ptr().add((j + 1) * n1 + i) as *mut f64, bot);
                    j += 2;
                }
                while j < j_end {
                    dst[j * n1 + i] = src[i * n2 + j];
                    dst[j * n1 + i + 1] = src[(i + 1) * n2 + j];
                    j += 1;
                }
                i += 2;
            }
            while i < i_end {
                for j in j_base..j_end {
                    dst[j * n1 + i] = src[i * n2 + j];
                }
                i += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn transpose_avx_reduced(src: &[Complex32], dst: &mut [Complex32], n1: usize, n2: usize) {
    use std::arch::x86_64::*;
    const TILE: usize = 16;
    for i_base in (0..n1).step_by(TILE) {
        for j_base in (0..n2).step_by(TILE) {
            let i_end = (i_base + TILE).min(n1);
            let j_end = (j_base + TILE).min(n2);
            let mut i = i_base;
            while i + 4 <= i_end {
                let mut j = j_base;
                while j + 4 <= j_end {
                    let r0 = _mm256_loadu_si256(src.as_ptr().add(i * n2 + j) as *const __m256i);
                    let r1 =
                        _mm256_loadu_si256(src.as_ptr().add((i + 1) * n2 + j) as *const __m256i);
                    let r2 =
                        _mm256_loadu_si256(src.as_ptr().add((i + 2) * n2 + j) as *const __m256i);
                    let r3 =
                        _mm256_loadu_si256(src.as_ptr().add((i + 3) * n2 + j) as *const __m256i);
                    let t0 = _mm256_unpacklo_epi64(r0, r1);
                    let t1 = _mm256_unpackhi_epi64(r0, r1);
                    let t2 = _mm256_unpacklo_epi64(r2, r3);
                    let t3 = _mm256_unpackhi_epi64(r2, r3);
                    let o0 = _mm256_permute2x128_si256(t0, t2, 0x20);
                    let o1 = _mm256_permute2x128_si256(t1, t3, 0x20);
                    let o2 = _mm256_permute2x128_si256(t0, t2, 0x31);
                    let o3 = _mm256_permute2x128_si256(t1, t3, 0x31);
                    _mm256_storeu_si256(dst.as_mut_ptr().add(j * n1 + i) as *mut __m256i, o0);
                    _mm256_storeu_si256(dst.as_mut_ptr().add((j + 1) * n1 + i) as *mut __m256i, o1);
                    _mm256_storeu_si256(dst.as_mut_ptr().add((j + 2) * n1 + i) as *mut __m256i, o2);
                    _mm256_storeu_si256(dst.as_mut_ptr().add((j + 3) * n1 + i) as *mut __m256i, o3);
                    j += 4;
                }
                while j < j_end {
                    for ii in 0..4 {
                        dst[j * n1 + i + ii] = src[(i + ii) * n2 + j];
                    }
                    j += 1;
                }
                i += 4;
            }
            while i < i_end {
                for j in j_base..j_end {
                    dst[j * n1 + i] = src[i * n2 + j];
                }
                i += 1;
            }
        }
    }
}

#[inline]
pub(crate) fn transpose_tiled_scalar<C: Copy>(src: &[C], dst: &mut [C], n1: usize, n2: usize) {
    const TILE: usize = 16;
    for i in (0..n1).step_by(TILE) {
        for j in (0..n2).step_by(TILE) {
            let i_end = (i + TILE).min(n1);
            let j_end = (j + TILE).min(n2);
            for r in i..i_end {
                let src_row = r * n2;
                for c in j..j_end {
                    dst[c * n1 + r] = src[src_row + c];
                }
            }
        }
    }
}

#[inline]
pub(super) fn transpose_matrix_precise(
    src: &[Complex64],
    dst: &mut [Complex64],
    n1: usize,
    n2: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS_AVX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *HAS_AVX.get_or_init(|| std::is_x86_feature_detected!("avx")) {
            // SAFETY: AVX confirmed at runtime.
            unsafe {
                return transpose_avx_precise(src, dst, n1, n2);
            }
        }
    }
    transpose_tiled_scalar(src, dst, n1, n2);
}

#[inline]
pub(super) fn transpose_matrix_reduced(
    src: &[Complex32],
    dst: &mut [Complex32],
    n1: usize,
    n2: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS_AVX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *HAS_AVX.get_or_init(|| std::is_x86_feature_detected!("avx")) {
            // SAFETY: AVX confirmed at runtime.
            unsafe {
                return transpose_avx_reduced(src, dst, n1, n2);
            }
        }
    }
    transpose_tiled_scalar(src, dst, n1, n2);
}
