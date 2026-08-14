//! `factorize_composite` support/rejection contract tests.

use crate::application::execution::kernel::radix_shape::factorize_composite;

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
                [2, 3, 4, 5, 7, 11, 13, 17, 23].contains(&r),
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
