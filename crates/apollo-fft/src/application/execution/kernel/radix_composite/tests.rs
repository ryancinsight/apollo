//! Tests for the mixed-radix composite Stockham FFT kernel.

use super::{forward_inplace_with_radices, inverse_inplace_unnorm_with_radices};
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::radix_shape::factorize_composite;
use num_complex::{Complex32, Complex64};

fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max)
}

fn forward_inplace_64(data: &mut [Complex64]) {
    let radices = factorize_composite(data.len()).expect("test length must be 2/3/5/7-smooth");
    forward_inplace_with_radices(data, &radices);
}

fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    let radices = factorize_composite(data.len()).expect("test length must be 2/3/5/7-smooth");
    inverse_inplace_unnorm_with_radices(data, &radices);
}

fn forward_inplace_32(data: &mut [Complex32]) {
    let radices = factorize_composite(data.len()).expect("test length must be 2/3/5/7-smooth");
    forward_inplace_with_radices(data, &radices);
}

// ── factorize_composite ───────────────────────────────────────────────────

#[test]
fn factorize_supported_sizes() {
    for &n in &[
        3usize,
        5,
        6,
        7,
        9,
        10,
        12,
        14,
        15,
        18,
        21,
        24,
        25,
        28,
        35,
        42,
        48,
        49,
        50,
        56,
        63,
        70,
        75,
        98,
        100,
        120,
        125,
        147,
        150,
        192,
        200,
        210,
        240,
        245,
        250,
        294,
        300,
        343,
        3430,
        375,
        384,
        392,
        450,
        500,
        588,
        600,
        686,
        700,
        750,
        784,
        864,
        900,
        980,
        1000,
        1200,
        1400,
        1470,
        1500,
        1960,
        2000,
        2400,
        2500,
        2940,
        3000,
        3430 * 2,
        3430 * 3,
        4000,
        4500,
        5000,
        6000,
        7000,
        7500,
        10000,
    ] {
        let result = factorize_composite(n);
        assert!(result.is_some(), "factorize_composite({n}) returned None");
        let radices = result.unwrap();
        assert_eq!(
            radices.iter().product::<usize>(),
            n,
            "factorize_composite({n}): product mismatch"
        );
        for &r in &radices {
            assert!(
                [2, 3, 4, 5, 7, 8].contains(&r),
                "factorize_composite({n}): unsupported radix {r}"
            );
        }
    }
}

#[test]
fn factorize_pow2_returns_none() {
    for exp in 1..=20u32 {
        let n = 1usize << exp;
        assert!(
            factorize_composite(n).is_none(),
            "factorize_composite({n}) should be None for pure-PoT"
        );
    }
}

#[test]
fn factorize_non_smooth_returns_none() {
    for &n in &[
        11usize, 13, 17, 19, 22, 23, 26, 29, 31, 33, 34, 38, 46, 58, 121, 143,
    ] {
        assert!(
            factorize_composite(n).is_none(),
            "factorize_composite({n}) should be None (has prime > 7)"
        );
    }
}

// ── forward + roundtrip correctness ──────────────────────────────────────

fn check_forward(n: usize, tol: f64) {
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.37).sin(), (k as f64 * 0.19).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut got = input.clone();
    forward_inplace_64(&mut got);
    let err = max_err(&got, &expected);
    assert!(
        err < tol,
        "forward N={n}: max_err={err:.2e} (tol={tol:.2e})"
    );
}

fn check_roundtrip(n: usize, tol: f64) {
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.53).cos(), (k as f64 * 0.27).sin()))
        .collect();
    let mut buf = input.clone();
    forward_inplace_64(&mut buf);
    inverse_inplace_unnorm_64(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / n as f64).collect();
    let err = max_err(&recovered, &input);
    assert!(
        err < tol,
        "roundtrip N={n}: max_err={err:.2e} (tol={tol:.2e})"
    );
}

fn check_inverse(n: usize, tol: f64) {
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.61).cos(), (k as f64 * 0.43).sin()))
        .collect();
    let expected_unnorm: Vec<Complex64> = dft_inverse(&input)
        .into_iter()
        .map(|x| x * n as f64)
        .collect();
    let mut got = input.clone();
    inverse_inplace_unnorm_64(&mut got);
    let err = max_err(&got, &expected_unnorm);
    assert!(
        err < tol,
        "inverse N={n}: max_err={err:.2e} (tol={tol:.2e})"
    );
}

#[test]
fn forward_n7() {
    check_forward(7, 1e-13);
}
#[test]
fn forward_n3() {
    check_forward(3, 1e-13);
}
#[test]
fn forward_n5() {
    check_forward(5, 1e-13);
}
#[test]
fn forward_n9() {
    check_forward(9, 1e-12);
}
#[test]
fn forward_n15() {
    check_forward(15, 1e-12);
}
#[test]
fn forward_n25() {
    check_forward(25, 1e-12);
}
#[test]
fn forward_n6() {
    check_forward(6, 1e-13);
}
#[test]
fn forward_n10() {
    check_forward(10, 1e-12);
}
#[test]
fn forward_n14() {
    check_forward(14, 1e-12);
}
#[test]
fn forward_n21() {
    check_forward(21, 1e-11);
}

#[test]
fn forward_n100() {
    check_forward(100, 1e-11);
}
#[test]
fn forward_n1000() {
    check_forward(1000, 1e-9);
}
#[test]
fn forward_n10000() {
    check_forward(10000, 1e-8);
}

#[test]
fn forward_n12() {
    check_forward(12, 1e-13);
}

#[test]
fn twiddle_cache_distinguishes_radix_order_for_same_length() {
    let input: Vec<Complex64> = (0..12)
        .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
        .collect();
    let expected = dft_forward(&input);

    let mut radix_3_4 = input.clone();
    forward_inplace_with_radices(&mut radix_3_4, &[3, 4]);
    assert!(
        max_err(&radix_3_4, &expected) < 1e-12,
        "radix [3,4] cache path must match direct DFT"
    );

    let mut radix_4_3 = input;
    forward_inplace_with_radices(&mut radix_4_3, &[4, 3]);
    assert!(
        max_err(&radix_4_3, &expected) < 1e-12,
        "radix [4,3] cache path must not reuse [3,4] twiddles"
    );
}

#[test]
fn forward_n24() {
    check_forward(24, 1e-12);
}
#[test]
fn forward_n48() {
    check_forward(48, 1e-12);
}
#[test]
fn forward_n192() {
    check_forward(192, 1e-11);
}
#[test]
fn forward_n384() {
    check_forward(384, 1e-10);
}

#[test]
fn roundtrip_n100() {
    check_roundtrip(100, 1e-12);
}
#[test]
fn roundtrip_n1000() {
    check_roundtrip(1000, 1e-11);
}
#[test]
fn roundtrip_n14() {
    check_roundtrip(14, 1e-12);
}
#[test]
fn roundtrip_n10000() {
    check_roundtrip(10000, 1e-10);
}

#[test]
fn inverse_n14() {
    check_inverse(14, 1e-12);
}
#[test]
fn inverse_n100() {
    check_inverse(100, 1e-11);
}
#[test]
fn inverse_n1000() {
    check_inverse(1000, 1e-10);
}

#[test]
fn forward_dc_n100() {
    let mut buf = vec![Complex64::new(1.0, 0.0); 100];
    forward_inplace_64(&mut buf);
    assert!((buf[0] - Complex64::new(100.0, 0.0)).norm() < 1e-10);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-10, "non-zero bin: {:?}", x);
    }
}

#[test]
fn forward_dc_n1000() {
    let mut buf = vec![Complex64::new(1.0, 0.0); 1000];
    forward_inplace_64(&mut buf);
    assert!((buf[0] - Complex64::new(1000.0, 0.0)).norm() < 1e-9);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-9, "non-zero bin: {:?}", x);
    }
}

#[test]
fn forward_f32_n100_matches_f64_reference() {
    let input: Vec<Complex64> = (0..100usize)
        .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.47).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: Vec<Complex32> = input
        .iter()
        .map(|x| Complex32::new(x.re as f32, x.im as f32))
        .collect();
    forward_inplace_32(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 1e-4, "f32 forward N=100 max_err={err:.2e}");
}

#[test]
fn forward_f32_n1000_matches_f64_reference() {
    let input: Vec<Complex64> = (0..1000usize)
        .map(|k| Complex64::new((k as f64 * 0.13).sin(), (k as f64 * 0.31).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: Vec<Complex32> = input
        .iter()
        .map(|x| Complex32::new(x.re as f32, x.im as f32))
        .collect();
    forward_inplace_32(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 2e-3, "f32 forward N=1000 max_err={err:.2e}");
}
