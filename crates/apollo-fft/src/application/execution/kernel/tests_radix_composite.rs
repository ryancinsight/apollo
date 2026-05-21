use super::*;
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
    let radices = factorize_composite(data.len()).expect("test length must be prime23-smooth");
    forward_inplace_with_radices(data, &radices);
}

fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    let radices = factorize_composite(data.len()).expect("test length must be prime23-smooth");
    inverse_inplace_unnorm_with_radices(data, &radices);
}

fn forward_inplace_32(data: &mut [Complex32]) {
    let radices = factorize_composite(data.len()).expect("test length must be prime23-smooth");
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
                [2, 3, 5, 7, 11, 13, 17, 23].contains(&r),
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
        19usize, 29, 31, 37, 38, 41, 43, 47, 53, 57, 58, 59, 61, 62, 74, 76,
    ] {
        assert!(
            factorize_composite(n).is_none(),
            "factorize_composite({n}) should be None (has prime > 23)"
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
    forward_inplace_with_radices(&mut radix_3_4, &[3, 2, 2]);
    assert!(
        max_err(&radix_3_4, &expected) < 1e-12,
        "radix [3,2,2] cache path must match direct DFT"
    );

    let mut radix_4_3 = input;
    forward_inplace_with_radices(&mut radix_4_3, &[2, 2, 3]);
    assert!(
        max_err(&radix_4_3, &expected) < 1e-12,
        "radix [2,2,3] cache path must not reuse [3,2,2] twiddles"
    );
}

#[test]
fn forward_lowered_radix4_tail_n12_matches_direct() {
    let input: Vec<Complex64> = (0..12)
        .map(|i| Complex64::new((i as f64 * 0.41).sin(), (i as f64 * 0.17).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut got = input;
    forward_inplace_with_radices(&mut got, &[3, 4]);
    assert!(
        max_err(&got, &expected) < 1e-12,
        "lowered radix [3,4] path must match direct DFT"
    );
}

#[test]
fn forward_lowered_radix4_tail_n192_matches_direct() {
    let input: Vec<Complex64> = (0..192)
        .map(|i| Complex64::new((i as f64 * 0.23).sin(), (i as f64 * 0.31).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut got = input;
    forward_inplace_with_radices(&mut got, &[3, 4, 4, 4]);
    assert!(
        max_err(&got, &expected) < 1e-10,
        "lowered radix [3,4,4,4] path must match direct DFT"
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
fn forward_reduced_n100_matches_precise_reference() {
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

// ── Extended prime radix coverage (11, 13, 17, 23) ──────────────────────

#[test]
fn forward_n11() {
    check_forward(11, 1e-13);
}
#[test]
fn forward_n22() {
    check_forward(22, 1e-13);
}
#[test]
fn forward_n33() {
    check_forward(33, 1e-12);
}
#[test]
fn forward_n34() {
    check_forward(34, 1e-12);
}
#[test]
fn forward_n46() {
    check_forward(46, 1e-12);
}
#[test]
fn forward_n121() {
    check_forward(121, 1e-11);
}
#[test]
fn forward_n143() {
    check_forward(143, 1e-11);
}
#[test]
fn forward_n352() {
    check_forward(352, 1e-10);
}
#[test]
fn roundtrip_n22() {
    check_roundtrip(22, 1e-13);
}
#[test]
fn roundtrip_n121() {
    check_roundtrip(121, 1e-11);
}
#[test]
fn roundtrip_n143() {
    check_roundtrip(143, 1e-11);
}
#[test]
fn roundtrip_n352() {
    check_roundtrip(352, 1e-10);
}
#[test]
fn inverse_n34() {
    check_inverse(34, 1e-12);
}
#[test]
fn inverse_n143() {
    check_inverse(143, 1e-11);
}

// ── 17 and 23 coverage ────────────────────────────────────────────────────

#[test]
fn forward_n17() {
    check_forward(17, 1e-13);
}
#[test]
fn forward_n23() {
    check_forward(23, 1e-13);
}
#[test]
fn forward_n34_via_17() {
    check_forward(34, 1e-12);
}
#[test]
fn forward_n46_via_23() {
    check_forward(46, 1e-12);
}
#[test]
fn forward_n51() {
    check_forward(51, 1e-12);
} // 17×3
#[test]
fn forward_n69() {
    check_forward(69, 1e-12);
} // 23×3
#[test]
fn forward_n242() {
    check_forward(242, 1e-10);
} // 11²×2
#[test]
fn forward_n264() {
    check_forward(264, 1e-10);
} // 11×3×8
#[test]
fn forward_n289() {
    check_forward(289, 1e-10);
} // 17²
#[test]
fn forward_n2200() {
    check_forward(2200, 1e-8);
} // 11×5²×8
#[test]
fn roundtrip_n242() {
    check_roundtrip(242, 1e-11);
}
#[test]
fn roundtrip_n2200() {
    check_roundtrip(2200, 1e-9);
}
#[test]
fn inverse_n242() {
    check_inverse(242, 1e-10);
}

// ── Three-odd-prime + radix-4 tail coverage ──────────────────────────────

#[test]
fn forward_n108() {
    check_forward(108, 1e-11);
} // 3³×4
#[test]
fn forward_n180() {
    check_forward(180, 1e-11);
} // 5×3²×4
#[test]
fn forward_n252() {
    check_forward(252, 1e-11);
} // 7×3²×4
#[test]
fn forward_n420() {
    check_forward(420, 1e-10);
} // 7×5×3×4
#[test]
fn roundtrip_n108() {
    check_roundtrip(108, 1e-11);
}
#[test]
fn roundtrip_n420() {
    check_roundtrip(420, 1e-10);
}
#[test]
fn inverse_n252() {
    check_inverse(252, 1e-11);
}

#[test]
fn forward_n36() {
    check_forward(36, 1e-12);
} // 3²×4
#[test]
fn forward_n60() {
    check_forward(60, 1e-12);
} // 5×3×4
#[test]
fn forward_n84() {
    check_forward(84, 1e-12);
} // 7×3×4
#[test]
fn forward_n140() {
    check_forward(140, 1e-11);
} // 7×5×4
#[test]
fn forward_n196() {
    check_forward(196, 1e-11);
} // 7²×4
#[test]
fn forward_n144() {
    check_forward(144, 1e-11);
} // 3²×16
#[test]
fn roundtrip_n60() {
    check_roundtrip(60, 1e-12);
}
#[test]
fn roundtrip_n144() {
    check_roundtrip(144, 1e-11);
}

#[test]
fn forward_reduced_n1000_matches_precise_reference() {
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
