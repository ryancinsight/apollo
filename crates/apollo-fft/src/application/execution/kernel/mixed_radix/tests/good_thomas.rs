use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::mixed_radix::{forward_inplace, inverse_inplace_unnorm};
use crate::application::execution::kernel::test_utils::{max_abs_err_32, max_abs_err_64};
use num_complex::{Complex32, Complex64};

#[test]
fn mixed_small_coprime_composites_use_short_codelets_and_match_direct() {
    for n in [6usize, 10, 12, 14] {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let mut forward = input.clone();
        forward_inplace::<f64>(&mut forward);
        let expected = dft_forward(&input);
        let err = max_abs_err_64(&forward, &expected);
        assert!(
            err < 1.0e-10,
            "small coprime forward N={n} mismatch err={err:.2e}"
        );

        let mut inverse = input.clone();
        inverse_inplace_unnorm::<f64>(&mut inverse);
        let expected = dft_inverse(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&inverse, &expected);
        assert!(
            err < 1.0e-10,
            "small coprime inverse N={n} mismatch err={err:.2e}"
        );
    }
}

#[test]
fn mixed_three_by_prime_composites_use_fused_crt_codelets_and_match_direct() {
    for n in [21usize, 33, 39] {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.23).sin(), (k as f64 * 0.31).cos()))
            .collect();
        let mut forward = input.clone();
        forward_inplace::<f64>(&mut forward);
        let expected = dft_forward(&input);
        let err = max_abs_err_64(&forward, &expected);
        assert!(
            err < 1.0e-10,
            "3-by-prime fused CRT forward N={n} mismatch err={err:.2e}"
        );

        let mut inverse = input.clone();
        inverse_inplace_unnorm::<f64>(&mut inverse);
        let expected = dft_inverse(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&inverse, &expected);
        assert!(
            err < 1.0e-10,
            "3-by-prime fused CRT inverse N={n} mismatch err={err:.2e}"
        );
    }
}

#[test]
fn mixed_three_by_prime_n33_matches_direct_for_f32() {
    let n = 33usize;
    let input: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.23).sin(), (k as f32 * 0.31).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f32>(&mut got);
    let expected = dft_forward(&input);
    let err = max_abs_err_32(&got, &expected);
    assert!(
        err < 3.0e-5,
        "3-by-prime fused CRT f32 forward N={n} mismatch err={err:.2e}"
    );
}

#[test]
fn mixed_new_short_good_thomas_codelets_match_direct() {
    for n in [18usize, 24, 36] {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.23).sin(), (k as f64 * 0.31).cos()))
            .collect();
        let mut forward = input.clone();
        forward_inplace::<f64>(&mut forward);
        let expected = dft_forward(&input);
        let err = max_abs_err_64(&forward, &expected);
        assert!(
            err < 1.0e-9,
            "Good-Thomas coprime forward N={n} mismatch err={err:.2e}"
        );

        let mut inverse = input.clone();
        inverse_inplace_unnorm::<f64>(&mut inverse);
        let expected = dft_inverse(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&inverse, &expected);
        assert!(
            err < 1.0e-9,
            "Good-Thomas coprime inverse N={n} mismatch err={err:.2e}"
        );
    }
}

#[test]
fn mixed_fixed_coprime_good_thomas_codelets_match_direct() {
    // NOTE: 2×prime sizes (22, 26, 34, 38, 46, 58, 62, 74, 82, 86, 94, 106) are
    // excluded — two_by_prime handles every prime×2 composite first in pfa_fft dispatch.
    for n in [
        28usize, 30, 35, 36, 42, 44, 48, 50, 52, 54, 68, 70, 72, 76, 77, 78, 80, 84, 88, 90, 92,
        99, 111, 120, 123, 129, 141, 143, 148, 150, 154, 159, 164, 165, 172, 175, 176, 180, 185,
        188, 198, 200,
    ] {
        eprintln!("Testing size {}", n);
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.29).cos()))
            .collect();
        let mut forward = input.clone();
        forward_inplace::<f64>(&mut forward);
        let expected = dft_forward(&input);
        let err = max_abs_err_64(&forward, &expected);
        assert!(
            err < 1.0e-9,
            "fixed Good-Thomas forward N={n} mismatch err={err:.2e}"
        );

        let mut inverse = input.clone();
        inverse_inplace_unnorm::<f64>(&mut inverse);
        let expected = dft_inverse(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&inverse, &expected);
        assert!(
            err < 1.0e-9,
            "fixed Good-Thomas inverse N={n} mismatch err={err:.2e}"
        );
    }
}

#[test]
fn mixed_fixed_good_thomas_n77_matches_direct_for_f32() {
    let n = 77usize;
    let input: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.17).sin(), (k as f32 * 0.29).cos()))
        .collect();
    let mut forward = input.clone();
    forward_inplace::<f32>(&mut forward);
    let expected = dft_forward(&input);
    let err = max_abs_err_32(&forward, &expected);
    assert!(
        err < 8.0e-5,
        "fixed Good-Thomas f32 forward N={n} mismatch err={err:.2e}"
    );
}
