use super::fixed::{cmul_pair32, cmul_vec32, store_complex32_low};
use num_complex::Complex32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stockham_quad_split_pair32(
    value: std::arch::x86_64::__m256,
    twiddle: Complex32,
) -> [std::arch::x86_64::__m128; 2] {
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_set1_ps, _mm_sub_ps,
    };

    let low = _mm256_castps256_ps128(value);
    let high = _mm256_extractf128_ps(value, 1);
    let product = cmul_pair32(_mm_set1_ps(twiddle.re), _mm_set1_ps(twiddle.im), high);
    [_mm_add_ps(low, product), _mm_sub_ps(low, product)]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stockham_quad_store_pair32(
    dst_ptr: *mut Complex32,
    low_index: usize,
    high_index: usize,
    pair: std::arch::x86_64::__m128,
    twiddle: Complex32,
) {
    use std::arch::x86_64::{_mm_add_ps, _mm_movehl_ps, _mm_set1_ps, _mm_sub_ps};

    let odd = _mm_movehl_ps(pair, pair);
    let product = cmul_pair32(_mm_set1_ps(twiddle.re), _mm_set1_ps(twiddle.im), odd);
    store_complex32_low(dst_ptr.add(low_index), _mm_add_ps(pair, product));
    store_complex32_low(dst_ptr.add(high_index), _mm_sub_ps(pair, product));
}

/// AVX/FMA radix-16 Stockham suffix for f64 when `groups == 8`.
///
/// The leaf evaluates four radix-2 Stockham stages for one digit `j` at a time.
/// The first three stages produce eight YMM pairs; the fourth stage consumes
/// each pair immediately through `stockham_quad_store_pair64`. This preserves
/// Stockham autosort order without a bit-reversal permutation and avoids
/// pairing adjacent digits, which measured as high register pressure in the
/// rejected contiguous-store candidate.
///
/// Algebraically, each stored band is `a ± W*b`, where `a` and `b` are the even
/// and odd outputs of the first three Stockham stages for the same digit. This
/// is exactly the fourth radix-2 Stockham recurrence, so direct substitution
/// proves equality with four scalar Stockham stages.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stockham_quad_groups_eight32(
    src: &[Complex32],
    dst: &mut [Complex32],
    radix: usize,
    first_twiddles: &[Complex32],
    second_twiddles: &[Complex32],
    third_twiddles: &[Complex32],
    fourth_twiddles: &[Complex32],
) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_sub_ps};

    let n = src.len();
    debug_assert_eq!(n, radix << 4);
    debug_assert_eq!(dst.len(), n);
    debug_assert!(first_twiddles.len() >= radix);
    debug_assert!(second_twiddles.len() >= radix << 1);
    debug_assert!(third_twiddles.len() >= radix << 2);
    debug_assert!(fourth_twiddles.len() >= radix << 3);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let first_ptr = first_twiddles.as_ptr();
    let second_ptr = second_twiddles.as_ptr();
    let third_ptr = third_twiddles.as_ptr();
    let fourth_ptr = fourth_twiddles.as_ptr();

    for j in 0..radix {
        let w1 = *first_ptr.add(j);
        let w20 = *second_ptr.add(j);
        let w21 = *second_ptr.add(j + radix);

        let w1r = _mm256_set1_ps(w1.re);
        let w1i = _mm256_set1_ps(w1.im);
        let w20r = _mm256_set1_ps(w20.re);
        let w20i = _mm256_set1_ps(w20.im);
        let w21r = _mm256_set1_ps(w21.re);
        let w21i = _mm256_set1_ps(w21.im);

        let src_base = j << 4;
        let x0123 = _mm256_loadu_ps(src_ptr.add(src_base).cast::<f32>());
        let x4567 = _mm256_loadu_ps(src_ptr.add(src_base + 4).cast::<f32>());
        let x89ab = cmul_vec32(
            w1r,
            w1i,
            _mm256_loadu_ps(src_ptr.add(src_base + 8).cast::<f32>()),
        );
        let xcdef = cmul_vec32(
            w1r,
            w1i,
            _mm256_loadu_ps(src_ptr.add(src_base + 12).cast::<f32>()),
        );

        let y0123 = _mm256_add_ps(x0123, x89ab);
        let y4567 = _mm256_add_ps(x4567, xcdef);
        let y89ab = _mm256_sub_ps(x0123, x89ab);
        let ycdef = _mm256_sub_ps(x4567, xcdef);

        let t4567 = cmul_vec32(w20r, w20i, y4567);
        let tcdef = cmul_vec32(w21r, w21i, ycdef);
        let z0123 = _mm256_add_ps(y0123, t4567);
        let z4567 = _mm256_add_ps(y89ab, tcdef);
        let z89ab = _mm256_sub_ps(y0123, t4567);
        let zcdef = _mm256_sub_ps(y89ab, tcdef);

        let [p01, p89] = stockham_quad_split_pair32(z0123, *third_ptr.add(j));
        let [p23, p10_11] = stockham_quad_split_pair32(z4567, *third_ptr.add(j + radix));
        let [p45, p12_13] = stockham_quad_split_pair32(z89ab, *third_ptr.add(j + 2 * radix));
        let [p67, p14_15] = stockham_quad_split_pair32(zcdef, *third_ptr.add(j + 3 * radix));

        stockham_quad_store_pair32(dst_ptr, j, j + 8 * radix, p01, *fourth_ptr.add(j));
        stockham_quad_store_pair32(
            dst_ptr,
            j + radix,
            j + 9 * radix,
            p23,
            *fourth_ptr.add(j + radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 2 * radix,
            j + 10 * radix,
            p45,
            *fourth_ptr.add(j + 2 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 3 * radix,
            j + 11 * radix,
            p67,
            *fourth_ptr.add(j + 3 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 4 * radix,
            j + 12 * radix,
            p89,
            *fourth_ptr.add(j + 4 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 5 * radix,
            j + 13 * radix,
            p10_11,
            *fourth_ptr.add(j + 5 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 6 * radix,
            j + 14 * radix,
            p12_13,
            *fourth_ptr.add(j + 6 * radix),
        );
        stockham_quad_store_pair32(
            dst_ptr,
            j + 7 * radix,
            j + 15 * radix,
            p14_15,
            *fourth_ptr.add(j + 7 * radix),
        );
    }
}
