//! Differential tests: every lane stage against its scalar recurrence.

use super::super::super::stage::stage_impl;
use super::super::stage::{stage_pair_impl, stage_triple_impl};
use super::{
    stage_groups_one_lanes, stage_lanes, stage_pair_groups_two_lanes, stage_pair_lanes,
    stage_pair_quarter_groups_two_lanes, stage_pair_radix_one_lanes,
    stage_triple_groups_eight_lanes, stage_triple_lanes, stage_triple_quarter_groups_one_lanes,
    stage_triple_radix_one_lanes,
};
use eunomia::Complex;

/// Deterministic samples in `[-1, 1]` from a 64-bit LCG (Knuth MMIX
/// constants), so a failure replays without a seed file.
fn samples<T: From<f32>>(count: usize, seed: u64) -> Vec<Complex<T>> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The top 24 bits give a uniform f32 in [0, 1); map to [-1, 1].
        ((state >> 40) as f32 / (1u32 << 24) as f32).mul_add(2.0, -1.0)
    };
    (0..count)
        .map(|_| Complex::new(T::from(next()), T::from(next())))
        .collect()
}

/// Unit-modulus twiddles `exp(-2πi k / (2 * radix))`, `count` of them.
fn twiddles<T: From<f32>>(radix: usize, count: usize) -> Vec<Complex<T>> {
    (0..count)
        .map(|k| {
            let angle = -core::f64::consts::TAU * k as f64 / (2 * radix) as f64;
            Complex::new(T::from(angle.cos() as f32), T::from(angle.sin() as f32))
        })
        .collect()
}

/// Each output is `a ± w·b` with `a`, `b` themselves `x ± w'·y`: at most
/// four unit-twiddled inputs, so `|out| ≤ 4·max|x| = 4`. The lane route
/// spends at most six roundings per output (two complex products, each
/// one product plus one FMA, and four adds) and the scalar route at most
/// eight; the routes' difference is bounded by the sum, 14 · ε · 4.
fn tolerance<T: Into<f64>>(epsilon: T) -> f64 {
    14.0 * 4.0 * epsilon.into()
}

fn assert_close<T: Copy + Into<f64>>(lanes: &[Complex<T>], scalar: &[Complex<T>], tol: f64) {
    for (index, (got, want)) in lanes.iter().zip(scalar).enumerate() {
        let dre = (got.re.into() - want.re.into()).abs();
        let dim = (got.im.into() - want.im.into()).abs();
        assert!(
            dre <= tol && dim <= tol,
            "sample {index}: lanes {:?} scalar {:?} (tolerance {tol:e})",
            (got.re.into(), got.im.into()),
            (want.re.into(), want.im.into())
        );
    }
}

/// `(n, radix)` pairs spanning the vector loop with and without a tail,
/// and the all-tail case (`half_groups < per_register` at both widths).
const GENERAL_CASES: &[(usize, usize)] = &[
    (16, 2),
    (32, 2),
    (64, 2),
    (64, 4),
    (128, 8),
    (1024, 2),
    (1024, 16),
    (1024, 128),
    (4096, 64),
];

const RADIX_ONE_SIZES: &[usize] = &[8, 16, 32, 64, 1024, 4096];

#[test]
fn general_stage_matches_scalar_recurrence_at_both_precisions() {
    for &(n, radix) in GENERAL_CASES {
        let src32 = samples::<f32>(n, 0x9E37_79B9_7F4A_7C15 ^ n as u64);
        let first32 = twiddles::<f32>(radix, radix);
        let second32 = twiddles::<f32>(2 * radix, 2 * radix);
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_pair_impl::<_, 1024>(&src32, &mut want32, radix, &first32, &second32);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(
            stage_pair_lanes::<f32, 8>(&src32, &mut got32, radix, &first32, &second32),
            "eight-lane f32 backend absent on this host"
        );
        assert_close(&got32, &want32, tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0xD1B5_4A32_D192_ED03 ^ n as u64);
        let first64 = twiddles::<f64>(radix, radix);
        let second64 = twiddles::<f64>(2 * radix, 2 * radix);
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_pair_impl::<_, 512>(&src64, &mut want64, radix, &first64, &second64);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(
            stage_pair_lanes::<f64, 4>(&src64, &mut got64, radix, &first64, &second64),
            "four-lane f64 backend absent on this host"
        );
        assert_close(&got64, &want64, tolerance(f64::EPSILON));

        // The AVX-512 width is host-dependent; when served it must agree too.
        let mut got512 = vec![Complex::new(0.0f64, 0.0); n];
        if stage_pair_lanes::<f64, 8>(&src64, &mut got512, radix, &first64, &second64) {
            assert_close(&got512, &want64, tolerance(f64::EPSILON));
        }
    }
}

#[test]
fn radix_one_stage_matches_scalar_recurrence_at_both_precisions() {
    for &n in RADIX_ONE_SIZES {
        let src32 = samples::<f32>(n, 0x2545_F491_4F6C_DD1D ^ n as u64);
        let second32 = twiddles::<f32>(2, 2);
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_pair_impl::<_, 1024>(&src32, &mut want32, 1, &second32, &second32);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(stage_pair_radix_one_lanes::<f32, 8>(
            &src32, &mut got32, &second32
        ));
        assert_close(&got32, &want32, tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0x853C_49E6_748F_EA9B ^ n as u64);
        let second64 = twiddles::<f64>(2, 2);
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_pair_impl::<_, 512>(&src64, &mut want64, 1, &second64, &second64);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(stage_pair_radix_one_lanes::<f64, 4>(
            &src64, &mut got64, &second64
        ));
        assert_close(&got64, &want64, tolerance(f64::EPSILON));
    }
}

#[test]
fn unserved_lane_width_touches_nothing() {
    let src = samples::<f32>(64, 7);
    let second = twiddles::<f32>(2, 2);
    let mut dst = vec![Complex::new(7.0f32, -7.0); 64];
    // No backend offers 64 f32 lanes; the caller keeps its scalar route.
    assert!(!stage_pair_radix_one_lanes::<f32, 64>(
        &src, &mut dst, &second
    ));
    assert!(dst.iter().all(|c| c.re == 7.0 && c.im == -7.0));
}

/// `(n, radix)` pairs for the single stage: the all-tail case at the f32
/// width (`groups = 2`), vector loops with and without a tail, and the
/// sized routes' group counts.
const BASE_CASES: &[(usize, usize)] = &[(8, 2), (16, 2), (32, 4), (64, 2), (1024, 8), (4096, 32)];

/// Each output is `a ± w·b` with unit `w`: at most two inputs, so
/// `|out| ≤ 2`. The lane route spends at most three roundings (one
/// product, one FMA, one add) and the scalar route at most four; the
/// routes' difference is bounded by the sum, 7 · ε · 2.
fn base_tolerance<T: Into<f64>>(epsilon: T) -> f64 {
    7.0 * 2.0 * epsilon.into()
}

#[test]
fn single_stage_matches_scalar_recurrence_at_both_precisions() {
    for &(n, radix) in BASE_CASES {
        let src32 = samples::<f32>(n, 0x5851_F42D_4C95_7F2D ^ n as u64);
        let tw32 = twiddles::<f32>(radix, radix);
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_impl::<_, 1024>(&src32, &mut want32, radix, &tw32);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(
            stage_lanes::<f32, 8>(&src32, &mut got32, radix, &tw32),
            "eight-lane f32 backend absent on this host"
        );
        assert_close(&got32, &want32, base_tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0x1405_7B7E_F767_814F ^ n as u64);
        let tw64 = twiddles::<f64>(radix, radix);
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_impl::<_, 512>(&src64, &mut want64, radix, &tw64);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(
            stage_lanes::<f64, 4>(&src64, &mut got64, radix, &tw64),
            "four-lane f64 backend absent on this host"
        );
        assert_close(&got64, &want64, base_tolerance(f64::EPSILON));

        let mut got512 = vec![Complex::new(0.0f64, 0.0); n];
        if stage_lanes::<f64, 8>(&src64, &mut got512, radix, &tw64) {
            assert_close(&got512, &want64, base_tolerance(f64::EPSILON));
        }
    }
}

/// `(n, radix)` pairs for the triple stage: `quarter_groups = 2` (all tail
/// at the f32 width, vector at f64), vector loops with and without a
/// tail, and the sized routes' group counts.
const TRIPLE_CASES: &[(usize, usize)] =
    &[(32, 2), (64, 2), (256, 4), (1024, 2), (1024, 8), (4096, 32)];

const TRIPLE_RADIX_ONE_SIZES: &[usize] = &[16, 32, 64, 256, 1024, 4096];

/// Each output combines eight inputs through three twiddled levels, so
/// `|out| ≤ 8`. Per level the lane route spends at most three roundings
/// (one product, one FMA, one add) and the scalar route at most four;
/// the routes' difference is bounded by the sum over three levels,
/// 21 · ε · 8.
fn triple_tolerance<T: Into<f64>>(epsilon: T) -> f64 {
    21.0 * 8.0 * epsilon.into()
}

#[test]
fn triple_stage_matches_scalar_recurrence_at_both_precisions() {
    for &(n, radix) in TRIPLE_CASES {
        let src32 = samples::<f32>(n, 0x7A2C_1E9B_3F5D_8A61 ^ n as u64);
        let (f32a, f32b, f32c) = (
            twiddles::<f32>(radix, radix),
            twiddles::<f32>(2 * radix, 2 * radix),
            twiddles::<f32>(4 * radix, 4 * radix),
        );
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_triple_impl::<_, 1024>(&src32, &mut want32, radix, &f32a, &f32b, &f32c);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(
            stage_triple_lanes::<f32, 8>(&src32, &mut got32, radix, &f32a, &f32b, &f32c),
            "eight-lane f32 backend absent on this host"
        );
        assert_close(&got32, &want32, triple_tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0x3C6E_F372_FE94_F82B ^ n as u64);
        let (f64a, f64b, f64c) = (
            twiddles::<f64>(radix, radix),
            twiddles::<f64>(2 * radix, 2 * radix),
            twiddles::<f64>(4 * radix, 4 * radix),
        );
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_triple_impl::<_, 512>(&src64, &mut want64, radix, &f64a, &f64b, &f64c);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(
            stage_triple_lanes::<f64, 4>(&src64, &mut got64, radix, &f64a, &f64b, &f64c),
            "four-lane f64 backend absent on this host"
        );
        assert_close(&got64, &want64, triple_tolerance(f64::EPSILON));

        let mut got512 = vec![Complex::new(0.0f64, 0.0); n];
        if stage_triple_lanes::<f64, 8>(&src64, &mut got512, radix, &f64a, &f64b, &f64c) {
            assert_close(&got512, &want64, triple_tolerance(f64::EPSILON));
        }
    }
}

#[test]
fn triple_radix_one_stage_matches_scalar_recurrence_at_both_precisions() {
    for &n in TRIPLE_RADIX_ONE_SIZES {
        // `second[1]` and `third[2]` are the exact quarter turns the
        // radix-one route rotates by; the scalar oracle multiplies by them.
        let src32 = samples::<f32>(n, 0x9E6C_63D0_676A_9A99 ^ n as u64);
        let (f32a, f32b, f32c) = (
            twiddles::<f32>(1, 1),
            twiddles::<f32>(2, 2),
            twiddles::<f32>(4, 4),
        );
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_triple_impl::<_, 1024>(&src32, &mut want32, 1, &f32a, &f32b, &f32c);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(stage_triple_radix_one_lanes::<f32, 8>(
            &src32, &mut got32, &f32b, &f32c
        ));
        assert_close(&got32, &want32, triple_tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0xB5AD_4ECE_DA1C_E2A9 ^ n as u64);
        let (f64a, f64b, f64c) = (
            twiddles::<f64>(1, 1),
            twiddles::<f64>(2, 2),
            twiddles::<f64>(4, 4),
        );
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_triple_impl::<_, 512>(&src64, &mut want64, 1, &f64a, &f64b, &f64c);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(stage_triple_radix_one_lanes::<f64, 4>(
            &src64, &mut got64, &f64b, &f64c
        ));
        assert_close(&got64, &want64, triple_tolerance(f64::EPSILON));
    }
}

/// Radices for the final stage (`n = 2 · radix`): the all-tail case at
/// the f32 width (`radix = 2`), vector loops with and without a tail,
/// and the sized routes' final-stage radices.
const GROUPS_ONE_RADICES: &[usize] = &[2, 4, 8, 16, 64, 512];

/// Each output is `a ± w·b`: one product (lane: product + FMA; scalar:
/// two products and an add) and one add — at most 3 and 4 roundings,
/// `|out| ≤ 2`, so `7 · ε · 2`.
#[test]
fn groups_one_stage_matches_scalar_recurrence_at_both_precisions() {
    for &radix in GROUPS_ONE_RADICES {
        let n = 2 * radix;
        let src32 = samples::<f32>(n, 0x2B99_2DDF_A232_49D6 ^ n as u64);
        let tw32 = twiddles::<f32>(radix, radix);
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_impl::<_, 1024>(&src32, &mut want32, radix, &tw32);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(stage_groups_one_lanes::<f32, 8>(
            &src32, &mut got32, radix, &tw32
        ));
        assert_close(&got32, &want32, base_tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0x6F4A_7F4F_E1D9_8B3C ^ n as u64);
        let tw64 = twiddles::<f64>(radix, radix);
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_impl::<_, 512>(&src64, &mut want64, radix, &tw64);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(stage_groups_one_lanes::<f64, 4>(
            &src64, &mut got64, radix, &tw64
        ));
        assert_close(&got64, &want64, base_tolerance(f64::EPSILON));

        let mut got512 = vec![Complex::new(0.0f64, 0.0); n];
        if stage_groups_one_lanes::<f64, 8>(&src64, &mut got512, radix, &tw64) {
            assert_close(&got512, &want64, base_tolerance(f64::EPSILON));
        }
    }
}

/// Radices for the `groups == 2` pair stage (`n = 4 · radix`): the
/// all-tail case at the f32 width (`radix = 2`), vector loops with and
/// without a tail, and the sized routes' radices.
const PAIR_GROUPS_TWO_RADICES: &[usize] = &[2, 4, 8, 16, 64, 256];

#[test]
fn pair_groups_two_stage_matches_scalar_recurrence_at_both_precisions() {
    for &radix in PAIR_GROUPS_TWO_RADICES {
        let n = 4 * radix;
        let src32 = samples::<f32>(n, 0x4A7F_1C3E_9B2D_6058 ^ n as u64);
        let first32 = twiddles::<f32>(radix, radix);
        let second32 = twiddles::<f32>(2 * radix, 2 * radix);
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_pair_impl::<_, 1024>(&src32, &mut want32, radix, &first32, &second32);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(stage_pair_groups_two_lanes::<f32, 8>(
            &src32, &mut got32, radix, &first32, &second32
        ));
        assert_close(&got32, &want32, tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0x8E1B_2C7A_5D93_F046 ^ n as u64);
        let first64 = twiddles::<f64>(radix, radix);
        let second64 = twiddles::<f64>(2 * radix, 2 * radix);
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_pair_impl::<_, 512>(&src64, &mut want64, radix, &first64, &second64);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(stage_pair_groups_two_lanes::<f64, 4>(
            &src64, &mut got64, radix, &first64, &second64
        ));
        assert_close(&got64, &want64, tolerance(f64::EPSILON));
    }
}

/// Radices for the `groups == 4` triple stage (`n = 8 · radix`): the
/// all-tail case at the f32 width (`radix = 2`), vector loops with and
/// without a tail, and the sized routes' radices.
const TRIPLE_QUARTER_GROUPS_ONE_RADICES: &[usize] = &[1, 2, 4, 8, 16, 128];

#[test]
fn triple_quarter_groups_one_stage_matches_scalar_recurrence_at_both_precisions() {
    for &radix in TRIPLE_QUARTER_GROUPS_ONE_RADICES {
        let n = 8 * radix;
        let src32 = samples::<f32>(n, 0xC0FF_EE11_2233_4455 ^ n as u64);
        let (f32a, f32b, f32c) = (
            twiddles::<f32>(radix, radix),
            twiddles::<f32>(2 * radix, 2 * radix),
            twiddles::<f32>(4 * radix, 4 * radix),
        );
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_triple_impl::<_, 1024>(&src32, &mut want32, radix, &f32a, &f32b, &f32c);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(stage_triple_quarter_groups_one_lanes::<f32, 8>(
            &src32, &mut got32, radix, &f32a, &f32b, &f32c
        ));
        assert_close(&got32, &want32, triple_tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0x1357_9BDF_2468_ACE0 ^ n as u64);
        let (f64a, f64b, f64c) = (
            twiddles::<f64>(radix, radix),
            twiddles::<f64>(2 * radix, 2 * radix),
            twiddles::<f64>(4 * radix, 4 * radix),
        );
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_triple_impl::<_, 512>(&src64, &mut want64, radix, &f64a, &f64b, &f64c);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(stage_triple_quarter_groups_one_lanes::<f64, 4>(
            &src64, &mut got64, radix, &f64a, &f64b, &f64c
        ));
        assert_close(&got64, &want64, triple_tolerance(f64::EPSILON));
    }
}

/// Radices for the two-digit stages: an odd count (all tail at the f32
/// width, vector at f64), vector loops with and without a tail, and the
/// sized routes' radices.
const TWO_DIGIT_RADICES: &[usize] = &[1, 2, 3, 4, 8, 16, 64];

#[test]
fn pair_quarter_groups_two_stage_matches_scalar_recurrence_at_both_precisions() {
    for &radix in TWO_DIGIT_RADICES {
        let n = 8 * radix;
        let src32 = samples::<f32>(n, 0x1A2B_3C4D_5E6F_7081 ^ n as u64);
        let first32 = twiddles::<f32>(radix, radix);
        let second32 = twiddles::<f32>(2 * radix, 2 * radix);
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_pair_impl::<_, 1024>(&src32, &mut want32, radix, &first32, &second32);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(stage_pair_quarter_groups_two_lanes::<f32, 8>(
            &src32, &mut got32, radix, &first32, &second32
        ));
        assert_close(&got32, &want32, tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0x9182_7364_5546_3728 ^ n as u64);
        let first64 = twiddles::<f64>(radix, radix);
        let second64 = twiddles::<f64>(2 * radix, 2 * radix);
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_pair_impl::<_, 512>(&src64, &mut want64, radix, &first64, &second64);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(stage_pair_quarter_groups_two_lanes::<f64, 4>(
            &src64, &mut got64, radix, &first64, &second64
        ));
        assert_close(&got64, &want64, tolerance(f64::EPSILON));
    }
}

#[test]
fn triple_groups_eight_stage_matches_scalar_recurrence_at_both_precisions() {
    for &radix in TWO_DIGIT_RADICES {
        let n = 16 * radix;
        let src32 = samples::<f32>(n, 0xF0E1_D2C3_B4A5_9687 ^ n as u64);
        let (f32a, f32b, f32c) = (
            twiddles::<f32>(radix, radix),
            twiddles::<f32>(2 * radix, 2 * radix),
            twiddles::<f32>(4 * radix, 4 * radix),
        );
        let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
        stage_triple_impl::<_, 1024>(&src32, &mut want32, radix, &f32a, &f32b, &f32c);
        let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
        assert!(stage_triple_groups_eight_lanes::<f32, 8>(
            &src32, &mut got32, radix, &f32a, &f32b, &f32c
        ));
        assert_close(&got32, &want32, triple_tolerance(f32::EPSILON));

        let src64 = samples::<f64>(n, 0x0F1E_2D3C_4B5A_6978 ^ n as u64);
        let (f64a, f64b, f64c) = (
            twiddles::<f64>(radix, radix),
            twiddles::<f64>(2 * radix, 2 * radix),
            twiddles::<f64>(4 * radix, 4 * radix),
        );
        let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
        stage_triple_impl::<_, 512>(&src64, &mut want64, radix, &f64a, &f64b, &f64c);
        let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
        assert!(stage_triple_groups_eight_lanes::<f64, 4>(
            &src64, &mut got64, radix, &f64a, &f64b, &f64c
        ));
        assert_close(&got64, &want64, triple_tolerance(f64::EPSILON));
    }
}
