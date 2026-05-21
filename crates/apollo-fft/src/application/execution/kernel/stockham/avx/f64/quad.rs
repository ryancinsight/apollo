use super::fixed::cmul_vec64;
use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
pub(crate) struct StockhamQuadFirstPairs64 {
    pub(crate) p01: std::arch::x86_64::__m256d,
    pub(crate) p45: std::arch::x86_64::__m256d,
    pub(crate) p89: std::arch::x86_64::__m256d,
    pub(crate) p12_13: std::arch::x86_64::__m256d,
}

#[cfg(target_arch = "x86_64")]
pub(crate) struct StockhamQuadSecondPairs64 {
    pub(crate) p23: std::arch::x86_64::__m256d,
    pub(crate) p67: std::arch::x86_64::__m256d,
    pub(crate) p10_11: std::arch::x86_64::__m256d,
    pub(crate) p14_15: std::arch::x86_64::__m256d,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stockham_quad_store_pair64(
    dst_ptr: *mut Complex64,
    low_index: usize,
    high_index: usize,
    pair: std::arch::x86_64::__m256d,
    twiddle: Complex64,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_castpd256_pd128, _mm256_permute2f128_pd, _mm256_set1_pd,
        _mm256_sub_pd, _mm_storeu_pd,
    };

    let odd = _mm256_permute2f128_pd(pair, pair, 0x11);
    let product = cmul_vec64(_mm256_set1_pd(twiddle.re), _mm256_set1_pd(twiddle.im), odd);
    _mm_storeu_pd(
        dst_ptr.add(low_index).cast::<f64>(),
        _mm256_castpd256_pd128(_mm256_add_pd(pair, product)),
    );
    _mm_storeu_pd(
        dst_ptr.add(high_index).cast::<f64>(),
        _mm256_castpd256_pd128(_mm256_sub_pd(pair, product)),
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stockham_quad_groups_eight64_first_pairs(
    src_ptr: *const Complex64,
    first_ptr: *const Complex64,
    second_ptr: *const Complex64,
    third_ptr: *const Complex64,
    radix: usize,
    j: usize,
) -> StockhamQuadFirstPairs64 {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_sub_pd};

    let w1 = *first_ptr.add(j);
    let w20 = *second_ptr.add(j);
    let w1r = _mm256_set1_pd(w1.re);
    let w1i = _mm256_set1_pd(w1.im);
    let w20r = _mm256_set1_pd(w20.re);
    let w20i = _mm256_set1_pd(w20.im);

    let src_base = j << 4;
    let x01 = _mm256_loadu_pd(src_ptr.add(src_base).cast::<f64>());
    let x23 = _mm256_loadu_pd(src_ptr.add(src_base + 2).cast::<f64>());
    let x45 = _mm256_loadu_pd(src_ptr.add(src_base + 4).cast::<f64>());
    let x67 = _mm256_loadu_pd(src_ptr.add(src_base + 6).cast::<f64>());
    let x89 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 8).cast::<f64>()),
    );
    let x10_11 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 10).cast::<f64>()),
    );
    let x12_13 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 12).cast::<f64>()),
    );
    let x14_15 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 14).cast::<f64>()),
    );

    let y01 = _mm256_add_pd(x01, x89);
    let y23 = _mm256_add_pd(x23, x10_11);
    let y45 = _mm256_add_pd(x45, x12_13);
    let y67 = _mm256_add_pd(x67, x14_15);
    let t45 = cmul_vec64(w20r, w20i, y45);
    let t67 = cmul_vec64(w20r, w20i, y67);

    let z01 = _mm256_add_pd(y01, t45);
    let z23 = _mm256_add_pd(y23, t67);
    let z89 = _mm256_sub_pd(y01, t45);
    let z10_11 = _mm256_sub_pd(y23, t67);

    let w30 = *third_ptr.add(j);
    let w30r = _mm256_set1_pd(w30.re);
    let w30i = _mm256_set1_pd(w30.im);
    let u23 = cmul_vec64(w30r, w30i, z23);
    let p01 = _mm256_add_pd(z01, u23);
    let p89 = _mm256_sub_pd(z01, u23);

    let w32 = *third_ptr.add(j + 2 * radix);
    let w32r = _mm256_set1_pd(w32.re);
    let w32i = _mm256_set1_pd(w32.im);
    let u10_11 = cmul_vec64(w32r, w32i, z10_11);
    let p45 = _mm256_add_pd(z89, u10_11);
    let p12_13 = _mm256_sub_pd(z89, u10_11);

    StockhamQuadFirstPairs64 {
        p01,
        p45,
        p89,
        p12_13,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stockham_quad_groups_eight64_second_pairs(
    src_ptr: *const Complex64,
    first_ptr: *const Complex64,
    second_ptr: *const Complex64,
    third_ptr: *const Complex64,
    radix: usize,
    j: usize,
) -> StockhamQuadSecondPairs64 {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_sub_pd};

    let w1 = *first_ptr.add(j);
    let w1r = _mm256_set1_pd(w1.re);
    let w1i = _mm256_set1_pd(w1.im);

    let src_base = j << 4;
    let x01 = _mm256_loadu_pd(src_ptr.add(src_base).cast::<f64>());
    let x23 = _mm256_loadu_pd(src_ptr.add(src_base + 2).cast::<f64>());
    let x45 = _mm256_loadu_pd(src_ptr.add(src_base + 4).cast::<f64>());
    let x67 = _mm256_loadu_pd(src_ptr.add(src_base + 6).cast::<f64>());
    let x89 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 8).cast::<f64>()),
    );
    let x10_11 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 10).cast::<f64>()),
    );
    let x12_13 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 12).cast::<f64>()),
    );
    let x14_15 = cmul_vec64(
        w1r,
        w1i,
        _mm256_loadu_pd(src_ptr.add(src_base + 14).cast::<f64>()),
    );

    let y89 = _mm256_sub_pd(x01, x89);
    let y10_11 = _mm256_sub_pd(x23, x10_11);
    let y12_13 = _mm256_sub_pd(x45, x12_13);
    let y14_15 = _mm256_sub_pd(x67, x14_15);

    let w21 = *second_ptr.add(j + radix);
    let w21r = _mm256_set1_pd(w21.re);
    let w21i = _mm256_set1_pd(w21.im);
    let t12_13 = cmul_vec64(w21r, w21i, y12_13);
    let t14_15 = cmul_vec64(w21r, w21i, y14_15);
    let z45 = _mm256_add_pd(y89, t12_13);
    let z67 = _mm256_add_pd(y10_11, t14_15);
    let z12_13 = _mm256_sub_pd(y89, t12_13);
    let z14_15 = _mm256_sub_pd(y10_11, t14_15);

    let w31 = *third_ptr.add(j + radix);
    let w31r = _mm256_set1_pd(w31.re);
    let w31i = _mm256_set1_pd(w31.im);
    let u67 = cmul_vec64(w31r, w31i, z67);
    let p23 = _mm256_add_pd(z45, u67);
    let p10_11 = _mm256_sub_pd(z45, u67);

    let w33 = *third_ptr.add(j + 3 * radix);
    let w33r = _mm256_set1_pd(w33.re);
    let w33i = _mm256_set1_pd(w33.im);
    let u14_15 = cmul_vec64(w33r, w33i, z14_15);
    let p67 = _mm256_add_pd(z12_13, u14_15);
    let p14_15 = _mm256_sub_pd(z12_13, u14_15);

    StockhamQuadSecondPairs64 {
        p23,
        p67,
        p10_11,
        p14_15,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn stockham_quad_groups_eight64_low_live(
    src: &[Complex64],
    dst: &mut [Complex64],
    radix: usize,
    first_twiddles: &[Complex64],
    second_twiddles: &[Complex64],
    third_twiddles: &[Complex64],
    fourth_twiddles: &[Complex64],
) {
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
        let first = stockham_quad_groups_eight64_first_pairs(
            src_ptr, first_ptr, second_ptr, third_ptr, radix, j,
        );
        stockham_quad_store_pair64(dst_ptr, j, j + 8 * radix, first.p01, *fourth_ptr.add(j));
        stockham_quad_store_pair64(
            dst_ptr,
            j + 2 * radix,
            j + 10 * radix,
            first.p45,
            *fourth_ptr.add(j + 2 * radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 4 * radix,
            j + 12 * radix,
            first.p89,
            *fourth_ptr.add(j + 4 * radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 6 * radix,
            j + 14 * radix,
            first.p12_13,
            *fourth_ptr.add(j + 6 * radix),
        );

        let second = stockham_quad_groups_eight64_second_pairs(
            src_ptr, first_ptr, second_ptr, third_ptr, radix, j,
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + radix,
            j + 9 * radix,
            second.p23,
            *fourth_ptr.add(j + radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 3 * radix,
            j + 11 * radix,
            second.p67,
            *fourth_ptr.add(j + 3 * radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 5 * radix,
            j + 13 * radix,
            second.p10_11,
            *fourth_ptr.add(j + 5 * radix),
        );
        stockham_quad_store_pair64(
            dst_ptr,
            j + 7 * radix,
            j + 15 * radix,
            second.p14_15,
            *fourth_ptr.add(j + 7 * radix),
        );
    }
}
