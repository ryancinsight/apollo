use super::super::{pfa_fft_natural_inplace, supports_ordered_rader_n1, ORDERED_RADER_SKIP_PRIMES};
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::test_utils::max_abs_err_64;
use eunomia::Complex64;
use proptest::prelude::*;
use proptest::proptest;

#[test]
fn ordered_rader_pfa_config_selects_rader_primes_only() {
    // Primes in ORDERED_RADER_SKIP_PRIMES must be rejected.
    for &n1 in &ORDERED_RADER_SKIP_PRIMES {
        assert!(
            !supports_ordered_rader_n1(n1),
            "n1={n1} (in ORDERED_RADER_SKIP_PRIMES) must be rejected"
        );
    }
    // Non-prime must be rejected.
    assert!(
        !supports_ordered_rader_n1(49),
        "non-prime n1=49 must be rejected"
    );

    // Sample of primes NOT in the skip set that should use ordered Rader.
    for n1 in [59usize, 61, 67, 71] {
        assert!(
            supports_ordered_rader_n1(n1),
            "n1={n1} should use ordered Rader inside PFA"
        );
    }
}

#[test]
fn natural_pfa_inplace_matches_direct_forward() {
    let n1 = 5usize;
    let n2 = 7usize;
    let n = n1 * n2;
    let input = signal(n, 0.19, 0.31);
    let mut got = input.clone();

    pfa_fft_natural_inplace::<f64, false>(&mut got, n1, n2);

    let expected = dft_forward(&input);
    let err = max_abs_err_64(&got, &expected);
    assert!(
        err < 1.0e-10,
        "in-place PFA forward N={n}, n1={n1}, n2={n2} mismatch err={err:.2e}"
    );
}

#[test]
fn natural_pfa_inplace_matches_direct_inverse_unnormalized() {
    let n1 = 5usize;
    let n2 = 7usize;
    let n = n1 * n2;
    let input = signal(n, 0.23, 0.17);
    let mut got = input.clone();

    pfa_fft_natural_inplace::<f64, true>(&mut got, n1, n2);

    let expected = dft_inverse(&input)
        .into_iter()
        .map(|x| x * n as f64)
        .collect::<Vec<_>>();
    let err = max_abs_err_64(&got, &expected);
    assert!(
        err < 1.0e-10,
        "in-place PFA inverse N={n}, n1={n1}, n2={n2} mismatch err={err:.2e}"
    );
}

#[test]
fn natural_pfa_inplace_matches_direct_forward_various_sizes() {
    // Test multiple coprime factor pairs to verify cycle-following correctness
    let cases = [
        (4usize, 5usize), // N=20
        (3usize, 7usize), // N=21
        (5usize, 6usize), // N=30
        (4usize, 9usize), // N=36
        (6usize, 7usize), // N=42
        (5usize, 8usize), // N=40
    ];
    for (n1, n2) in cases {
        let n = n1 * n2;
        let input = signal(n, 0.19, 0.31);
        let mut got = input.clone();

        pfa_fft_natural_inplace::<f64, false>(&mut got, n1, n2);

        let expected = dft_forward(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(
            err < 1.0e-10,
            "in-place PFA forward N={n}, n1={n1}, n2={n2} mismatch err={err:.2e}"
        );
    }
}

#[test]
fn fixed_exclude_primes_match_canonical_source_arrays() {
    // Verify FIXED_EXCLUDE_PRIMES in fixed.rs is the union of
    // two_by_prime::DIRECT_PAIR_PRIMES and three_by_prime::THREE_BY_PRIME_PRIMES.
    let mut expected: Vec<usize> = super::super::two_by_prime::DIRECT_PAIR_PRIMES
        .iter()
        .chain(super::super::three_by_prime::THREE_BY_PRIME_PRIMES)
        .copied()
        .collect();
    expected.sort_unstable();
    expected.dedup();

    let mut actual: Vec<usize> = super::super::fixed::FIXED_EXCLUDE_PRIMES.to_vec();
    actual.sort_unstable();
    actual.dedup();

    assert_eq!(
        actual, expected,
        "fixed::FIXED_EXCLUDE_PRIMES must match two_by_prime::DIRECT_PAIR_PRIMES \
         ∪ three_by_prime::THREE_BY_PRIME_PRIMES"
    );
}

#[test]
fn two_by_prime_direct_pair_const_matches_dispatch() {
    // Every prime in DIRECT_PAIR_PRIMES must be accepted by
    // direct_pair_prime().
    for &p in super::super::two_by_prime::DIRECT_PAIR_PRIMES {
        assert!(
            super::super::two_by_prime::direct_pair_prime(p),
            "{p} in DIRECT_PAIR_PRIMES must be accepted by direct_pair_prime()"
        );
    }
    // Primes NOT in DIRECT_PAIR_PRIMES must be rejected.
    for p in [2usize, 3, 5, 7, 59, 61] {
        assert!(
            !super::super::two_by_prime::direct_pair_prime(p),
            "{p} not in DIRECT_PAIR_PRIMES must be rejected by direct_pair_prime()"
        );
    }
}

fn signal(n: usize, real_step: f64, imag_step: f64) -> Vec<Complex64> {
    (0..n)
        .map(|k| Complex64::new((k as f64 * real_step).sin(), (k as f64 * imag_step).cos()))
        .collect()
}

proptest! {
    #[test]
    fn pfa_fft_correctness_proptest(
        raw_re in prop::collection::vec(-10.0f64..10.0f64, 150),
        raw_im in prop::collection::vec(-10.0f64..10.0f64, 150),
    ) {
        let check_pfa = |n1: usize, n2: usize| {
            let n = n1 * n2;
            let mut input = Vec::with_capacity(n);
            for i in 0..n {
                input.push(Complex64::new(raw_re[i], raw_im[i]));
            }

            // 1. Forward test
            let expected_fwd = dft_forward(&input);
            let mut got_fwd = input.clone();
            super::super::pfa_fft::<f64, false>(&mut got_fwd, n1, n2);
            let err_fwd = max_abs_err_64(&got_fwd, &expected_fwd);
            prop_assert!(err_fwd < 1.0e-9, "Forward mismatch at n1={}, n2={}, err={:.2e}", n1, n2, err_fwd);

            // 2. Inverse test
            let expected_inv: Vec<_> = dft_inverse(&input).into_iter().map(|x| x * n as f64).collect();
            let mut got_inv = input.clone();
            super::super::pfa_fft::<f64, true>(&mut got_inv, n1, n2);
            let err_inv = max_abs_err_64(&got_inv, &expected_inv);
            prop_assert!(err_inv < 1.0e-9, "Inverse mismatch at n1={}, n2={}, err={:.2e}", n1, n2, err_inv);

            Ok(())
        };

        // Test Two-by-Prime sizes
        for &p in super::super::two_by_prime::DIRECT_PAIR_PRIMES {
            if 2 * p <= 150 {
                check_pfa(2, p)?;
                check_pfa(p, 2)?;
            }
        }

        // Test Three-by-Prime sizes
        for &p in super::super::three_by_prime::THREE_BY_PRIME_PRIMES {
            if 3 * p <= 150 {
                check_pfa(3, p)?;
                check_pfa(p, 3)?;
            }
        }

        // Test Cook-Toom-GT sizes
        check_pfa(4, 15)?;
        check_pfa(15, 4)?;
        check_pfa(4, 21)?;
        check_pfa(21, 4)?;
        check_pfa(9, 10)?;
        check_pfa(10, 9)?;
        check_pfa(6, 25)?;
        check_pfa(25, 6)?;

        // Test standard PFA natural/ordered Rader sizes
        check_pfa(5, 7)?;
        check_pfa(7, 5)?;
        check_pfa(5, 8)?;
        check_pfa(8, 5)?;
    }
}
