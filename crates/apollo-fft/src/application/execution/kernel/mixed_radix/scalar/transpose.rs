use eunomia::{Complex32, Complex64};

#[cfg(test)]
mod tests;

/// Restrict both borrows to the matrix before any destination write. Standard
/// slice diagnostics report the required extent and offending buffer length.
#[track_caller]
fn matrix_slices<'source, 'destination, C>(
    source: &'source [C],
    destination: &'destination mut [C],
    rows: usize,
    columns: usize,
) -> (&'source [C], &'destination mut [C]) {
    let extent = rows
        .checked_mul(columns)
        .unwrap_or_else(|| panic!("transpose {rows} by {columns} has an unrepresentable extent"));
    (&source[..extent], &mut destination[..extent])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
/// # Safety
/// AVX is available, dimensions are nonzero, and their product fits both slices.
unsafe fn transpose_avx_precise(src: &[Complex64], dst: &mut [Complex64], n1: usize, n2: usize) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_permute2f128_pd, _mm256_storeu_pd};
    debug_assert!(n1 != 0 && n2 != 0);
    debug_assert!(n1 * n2 <= src.len() && n1 * n2 <= dst.len());
    const TILE: usize = 16;
    for i_base in (0..n1).step_by(TILE) {
        for j_base in (0..n2).step_by(TILE) {
            let i_end = (i_base + TILE).min(n1);
            let j_end = (j_base + TILE).min(n2);
            let mut i = i_base;
            while i + 2 <= i_end {
                let mut j = j_base;
                while j + 2 <= j_end {
                    // SAFETY: the validated extent and full two-by-two tile
                    // bound both row loads and transposed stores. Each load
                    // moves two repr(C) complex values; unaligned access is
                    // permitted, and the mutable destination is disjoint.
                    unsafe {
                        let r0 = _mm256_loadu_pd(src.as_ptr().add(i * n2 + j).cast::<f64>());
                        let r1 = _mm256_loadu_pd(src.as_ptr().add((i + 1) * n2 + j).cast::<f64>());
                        let top = _mm256_permute2f128_pd(r0, r1, 0x20);
                        let bot = _mm256_permute2f128_pd(r0, r1, 0x31);
                        _mm256_storeu_pd(dst.as_mut_ptr().add(j * n1 + i).cast::<f64>(), top);
                        _mm256_storeu_pd(dst.as_mut_ptr().add((j + 1) * n1 + i).cast::<f64>(), bot);
                    }
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
/// # Safety
/// AVX is available, dimensions are nonzero, and their product fits both slices.
unsafe fn transpose_avx_reduced(src: &[Complex32], dst: &mut [Complex32], n1: usize, n2: usize) {
    use std::arch::x86_64::{
        _mm256_loadu_pd, _mm256_permute2f128_pd, _mm256_storeu_pd, _mm256_unpackhi_pd,
        _mm256_unpacklo_pd,
    };
    debug_assert!(n1 != 0 && n2 != 0);
    debug_assert!(n1 * n2 <= src.len() && n1 * n2 <= dst.len());
    const TILE: usize = 16;
    for i_base in (0..n1).step_by(TILE) {
        for j_base in (0..n2).step_by(TILE) {
            let i_end = (i_base + TILE).min(n1);
            let j_end = (j_base + TILE).min(n2);
            let mut i = i_base;
            while i + 4 <= i_end {
                let mut j = j_base;
                while j + 4 <= j_end {
                    // One 64-bit lane holds the bits of one complex value.
                    // AVX moves/shuffles preserve those bits, including NaN
                    // payloads; no floating-point arithmetic or conversion
                    // occurs. Integer unpack/permute intrinsics require AVX2.
                    // SAFETY: the validated extent and full four-by-four
                    // tile bound four 32-byte row loads and disjoint stores.
                    // Complex32 is repr(C), eight bytes; unaligned access is
                    // permitted and source/destination borrows are disjoint.
                    unsafe {
                        let r0 = _mm256_loadu_pd(src.as_ptr().add(i * n2 + j).cast::<f64>());
                        let r1 = _mm256_loadu_pd(src.as_ptr().add((i + 1) * n2 + j).cast::<f64>());
                        let r2 = _mm256_loadu_pd(src.as_ptr().add((i + 2) * n2 + j).cast::<f64>());
                        let r3 = _mm256_loadu_pd(src.as_ptr().add((i + 3) * n2 + j).cast::<f64>());
                        let t0 = _mm256_unpacklo_pd(r0, r1);
                        let t1 = _mm256_unpackhi_pd(r0, r1);
                        let t2 = _mm256_unpacklo_pd(r2, r3);
                        let t3 = _mm256_unpackhi_pd(r2, r3);
                        let o0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                        let o1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                        let o2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                        let o3 = _mm256_permute2f128_pd(t1, t3, 0x31);
                        _mm256_storeu_pd(dst.as_mut_ptr().add(j * n1 + i).cast::<f64>(), o0);
                        _mm256_storeu_pd(dst.as_mut_ptr().add((j + 1) * n1 + i).cast::<f64>(), o1);
                        _mm256_storeu_pd(dst.as_mut_ptr().add((j + 2) * n1 + i).cast::<f64>(), o2);
                        _mm256_storeu_pd(dst.as_mut_ptr().add((j + 3) * n1 + i).cast::<f64>(), o3);
                    }
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
    let (src, dst) = matrix_slices(src, dst, n1, n2);
    if src.is_empty() {
        return;
    }
    const TILE: usize = 16;
    for i in (0..n1).step_by(TILE) {
        for j in (0..n2).step_by(TILE) {
            let i_end = i + (n1 - i).min(TILE);
            let j_end = j + (n2 - j).min(TILE);
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
        if std::is_x86_feature_detected!("avx") {
            let (src, dst) = matrix_slices(src, dst, n1, n2);
            if src.is_empty() {
                return;
            }
            // SAFETY: AVX is confirmed and the checked nonzero extent fits
            // both slices. Validation precedes every destination write.
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
        if std::is_x86_feature_detected!("avx") {
            let (src, dst) = matrix_slices(src, dst, n1, n2);
            if src.is_empty() {
                return;
            }
            // SAFETY: AVX is confirmed and the checked nonzero extent fits
            // both slices. Validation precedes every destination write.
            unsafe {
                return transpose_avx_reduced(src, dst, n1, n2);
            }
        }
    }
    transpose_tiled_scalar(src, dst, n1, n2);
}
