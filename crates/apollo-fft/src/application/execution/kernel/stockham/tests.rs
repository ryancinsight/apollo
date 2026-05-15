//! Stockham unit and differential tests.
#![allow(unused_imports)]

use super::butterfly::{
    build_butterfly512_twiddles_32, build_butterfly512_twiddles_64, hybrid_radix8x512_32_avx_fma,
    hybrid_radix8x512_64_avx_fma, stage_pair_impl, stage_quad_impl, stage_triple_impl,
    stockham_mixed_twiddle_32, stockham_mixed_twiddle_64,
};
use super::precision::{
    F32Stockham, F32StockhamAvxFma, F64Stockham, F64StockhamAvxFma, StockhamPrecision,
};
use super::*;
use crate::application::execution::kernel::stockham::avx::f64::triple_2::stage_triple64_groups_eight_avx_fma;
use num_complex::{Complex32, Complex64};

#[cfg(target_arch = "x86_64")]
#[test]
fn f32_avx_groups_eight_quad_stage_matches_scalar_reference() {
    if !std::arch::is_x86_feature_detected!("avx") || !std::arch::is_x86_feature_detected!("fma") {
        return;
    }

    let radix = 64usize;
    let n = radix << 4;
    let input: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.013).sin(), (k as f32 * 0.019).cos()))
        .collect();
    let twiddles =
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(n);
    let base = radix - 1;
    let first = &twiddles[base..base + radix];
    let second = &twiddles[base + radix..base + 3 * radix];
    let third = &twiddles[base + 3 * radix..base + 7 * radix];
    let fourth = &twiddles[base + 7 * radix..base + 15 * radix];
    let mut expected = vec![Complex32::new(0.0, 0.0); n];
    let mut actual = expected.clone();

    stage_quad_impl(&input, &mut expected, radix, first, second, third, fourth);
    <F32StockhamAvxFma as StockhamPrecision>::stage_quad(
        &input,
        &mut actual,
        radix,
        first,
        second,
        third,
        fourth,
    );

    let err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f32::max);
    assert!(
        err < 1.0e-4,
        "groups-eight f32 AVX quad stage err={err:.2e}"
    );
}

#[test]
fn scalar_fallback_policy_types_remain_reachable_in_tests() {
    let _ = F64Stockham;
    let _ = F32Stockham;
    assert_eq!(<F64Stockham as StockhamPrecision>::MAX_FUSED_STAGES, 4);
    assert_eq!(<F32Stockham as StockhamPrecision>::MAX_FUSED_STAGES, 4);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn stockham_scheduler_uses_copyback_instead_of_stride1_prepass() {
    let source = include_str!("transform.rs");
    let body = source
        .split_once("fn transform<P: StockhamPrecision>(")
        .map(|(_, tail)| tail)
        .expect("generic Stockham transform body must be present");
    assert!(!body.contains("schedule_odd_flips::<P>"));
    assert!(!body.contains("prepass_twiddles"));
    assert!(body.contains("data.copy_from_slice(scratch);"));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn butterfly512_f32_packed_twiddles_match_separated_column_contract() {
    let twiddles =
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(512);
    let packed = build_butterfly512_twiddles_32(&twiddles);

    assert_eq!(packed.len(), 120);
    for columnset in 0..8 {
        let col_base = columnset * 4;
        for row in 1..16 {
            let vector = packed[columnset * 15 + row - 1];
            for lane in 0..4 {
                let expected = stockham_mixed_twiddle_32::<16, 32>(&twiddles, row, col_base + lane);
                assert_eq!(
                    vector[lane],
                    expected,
                    "f32 packed twiddle row={row} col={}",
                    col_base + lane
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn butterfly512_f64_packed_twiddles_match_separated_column_contract() {
    let twiddles =
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(512);
    let packed = build_butterfly512_twiddles_64(&twiddles);

    assert_eq!(packed.len(), 240);
    for columnset in 0..16 {
        let col_base = columnset * 2;
        for row in 1..16 {
            let vector = packed[columnset * 15 + row - 1];
            for lane in 0..2 {
                let expected = stockham_mixed_twiddle_64::<16, 32>(&twiddles, row, col_base + lane);
                assert_eq!(
                    vector[lane],
                    expected,
                    "f64 packed twiddle row={row} col={}",
                    col_base + lane
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx_scheduler_selects_f32_n4096_tail_shape() {
    assert!(<F64StockhamAvxFma as StockhamPrecision>::stage_triple_enabled(64, 4096, true));
    assert!(!<F64StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(256, 4096, false));
    assert!(!<F64StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(64, 1024, true));

    assert!(<F32StockhamAvxFma as StockhamPrecision>::stage_triple_enabled(64, 4096, true));
    assert!(!<F32StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(256, 4096, false));
    assert!(!<F32StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(64, 1024, true));
    assert!(<F32StockhamAvxFma as StockhamPrecision>::stage_triple_enabled(512, 8192, false));
    assert!(!<F32StockhamAvxFma as StockhamPrecision>::stage_quad_enabled(512, 8192, false));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn f64_triple_avx_routes_groups_eight_to_dedicated_late_leaf() {
    let source = include_str!("precision/f64_impl.rs");
    let body = source
        .split_once("impl StockhamPrecision for F64StockhamAvxFma")
        .map(|(_, tail)| tail)
        .expect("F64StockhamAvxFma implementation must be present");

    assert!(body.contains("groups == 8"));
    assert!(body.contains("stage_triple64_groups_eight_avx_fma("));
    assert!(!body.contains("bit_reverse"));
    assert!(!body.contains("reverse_bits"));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn f64_avx_groups_eight_triple_stage_matches_scalar_reference() {
    if !std::arch::is_x86_feature_detected!("avx") || !std::arch::is_x86_feature_detected!("fma") {
        return;
    }

    let radix = 64usize;
    let n = radix << 4;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.013).sin(), (k as f64 * 0.019).cos()))
        .collect();
    let twiddles =
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(n);
    let base = radix - 1;
    let first = &twiddles[base..base + radix];
    let second = &twiddles[base + radix..base + 3 * radix];
    let third = &twiddles[base + 3 * radix..base + 7 * radix];
    let mut expected = vec![Complex64::new(0.0, 0.0); n];
    let mut actual = expected.clone();

    stage_triple_impl(&input, &mut expected, radix, first, second, third);
    unsafe {
        stage_triple64_groups_eight_avx_fma(&input, &mut actual, radix, first, second, third)
    };

    let err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(
        err < 1.0e-12,
        "groups-eight f64 AVX triple stage err={err:.2e}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn f64_hybrid_radix8x512_matches_stockham_n4096() {
    if !std::arch::is_x86_feature_detected!("avx") || !std::arch::is_x86_feature_detected!("fma") {
        return;
    }

    let n = 4096usize;
    let mut expected: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.007).sin(), (k as f64 * 0.011).cos()))
        .collect();
    let mut actual = expected.clone();
    let twiddles =
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(n);
    let mut expected_scratch = vec![Complex64::new(0.0, 0.0); n];
    let mut actual_scratch = vec![Complex64::new(0.0, 0.0); n];

    transform::transform::<F64StockhamAvxFma>(
        &mut expected,
        &mut expected_scratch,
        &twiddles,
        None,
    );
    unsafe {
        hybrid_radix8x512_64_avx_fma::<false>(&mut actual, &mut actual_scratch, &twiddles);
    }

    let err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(err < 1.0e-10, "f64 hybrid radix8x512 err={err:.2e}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn f32_hybrid_radix8x512_matches_stockham_n4096() {
    if !std::arch::is_x86_feature_detected!("avx") || !std::arch::is_x86_feature_detected!("fma") {
        return;
    }

    let n = 4096usize;
    let mut expected: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.007).sin(), (k as f32 * 0.011).cos()))
        .collect();
    let mut actual = expected.clone();
    let twiddles =
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(n);
    let mut expected_scratch = vec![Complex32::new(0.0, 0.0); n];
    let mut actual_scratch = vec![Complex32::new(0.0, 0.0); n];

    transform::transform::<F32StockhamAvxFma>(
        &mut expected,
        &mut expected_scratch,
        &twiddles,
        None,
    );
    unsafe {
        hybrid_radix8x512_32_avx_fma::<false>(&mut actual, &mut actual_scratch, &twiddles);
    }

    let err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f32::max);
    let tolerance = (n as f32 / 2.0) * f32::EPSILON;
    assert!(
        err < tolerance,
        "f32 hybrid radix8x512 err={err:.2e} tolerance={tolerance:.2e}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn f64_hybrid_radix8x512_inverse_roundtrip_n4096() {
    if !std::arch::is_x86_feature_detected!("avx") || !std::arch::is_x86_feature_detected!("fma") {
        return;
    }

    let n = 4096usize;
    let mut data: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.017).sin(), (k as f64 * 0.023).cos()))
        .collect();
    let original = data.clone();
    let forward =
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(n);
    let inverse =
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_inverse_twiddle_table(n);
    let mut scratch = vec![Complex64::new(0.0, 0.0); n];

    unsafe {
        hybrid_radix8x512_64_avx_fma::<false>(&mut data, &mut scratch, &forward);
        hybrid_radix8x512_64_avx_fma::<true>(&mut data, &mut scratch, &inverse);
    }
    data.iter_mut().for_each(|value| *value *= 1.0 / n as f64);

    let err = data
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(err < 1.0e-10, "f64 hybrid inverse roundtrip err={err:.2e}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn f32_hybrid_radix8x512_inverse_roundtrip_n4096() {
    if !std::arch::is_x86_feature_detected!("avx") || !std::arch::is_x86_feature_detected!("fma") {
        return;
    }

    let n = 4096usize;
    let mut data: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.017).sin(), (k as f32 * 0.023).cos()))
        .collect();
    let original = data.clone();
    let forward =
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(n);
    let inverse =
        <f32 as crate::application::execution::kernel::real_fft::RealFft>::build_inverse_twiddle_table(n);
    let mut scratch = vec![Complex32::new(0.0, 0.0); n];

    unsafe {
        hybrid_radix8x512_32_avx_fma::<false>(&mut data, &mut scratch, &forward);
        hybrid_radix8x512_32_avx_fma::<true>(&mut data, &mut scratch, &inverse);
    }
    data.iter_mut().for_each(|value| *value *= 1.0 / n as f32);

    let err = data
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f32::max);
    assert!(err < 1.0e-4, "f32 hybrid inverse roundtrip err={err:.2e}");
}

#[test]
fn hybrid_radix8x512_source_has_no_bit_reversal_or_allocation() {
    let source = include_str!("butterfly/hybrid.rs");
    let body = source
        .split_once("unsafe fn hybrid_radix8x512_64_avx_fma")
        .map(|(_, tail)| tail)
        .expect("hybrid radix8x512 body must be present");

    assert!(!body.contains("bit_reverse"));
    assert!(!body.contains("reverse_bits"));
    assert!(!body.contains("bitrev"));
    assert!(!body.contains("Vec<"));
    assert!(!body.contains("vec!"));
    assert!(!body.contains("Box<"));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn f64_avx_schedule_roundtrip_holds_for_n8192() {
    if !std::arch::is_x86_feature_detected!("avx") || !std::arch::is_x86_feature_detected!("fma") {
        return;
    }

    let mut data: Vec<Complex64> = (0..8192)
        .map(|k| Complex64::new((k as f64 * 0.007).sin(), (k as f64 * 0.011).cos()))
        .collect();
    let original = data.clone();
    let mut scratch = vec![Complex64::new(0.0, 0.0); data.len()];
    let forward =
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_forward_twiddle_table(data.len());
    let inverse =
        <f64 as crate::application::execution::kernel::real_fft::RealFft>::build_inverse_twiddle_table(data.len());

    f64::forward_with_scratch(&mut data, &mut scratch, &forward);
    // inverse_with_scratch removed: implement as forward on inverse twiddles + 1/N scale.
    f64::forward_with_scratch(&mut data, &mut scratch, &inverse);
    let scale = 1.0 / data.len() as f64;
    for v in &mut data {
        *v *= scale;
    }

    let err = data
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(
        err < 1.0e-10,
        "n8192 f64 AVX Stockham roundtrip err={err:.2e}"
    );
}
