use super::super::cook_toom_gt::{dft150_impl, dft60_impl, dft84_impl, dft90_impl, try_fft};
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::test_utils::max_abs_err_64;
use num_complex::Complex64;

fn signal84() -> Vec<Complex64> {
    (0..84)
        .map(|k| Complex64::new((k as f64 * 0.19).sin(), (k as f64 * 0.31).cos()))
        .collect()
}

fn signal60() -> Vec<Complex64> {
    (0..60)
        .map(|k| Complex64::new((k as f64 * 0.23).sin(), (k as f64 * 0.37).cos()))
        .collect()
}

#[test]
fn dft84_forward_matches_direct() {
    let input = signal84();
    let expected = dft_forward(&input);
    let mut got = input.clone();
    dft84_impl::<f64, false>(&mut got);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft84 forward mismatch err={}", err);
}

#[test]
fn dft84_inverse_matches_direct_unnormalized() {
    let input = signal84();
    let expected: Vec<_> = dft_inverse(&input).into_iter().map(|x| x * 84.0).collect();
    let mut got = input.clone();
    dft84_impl::<f64, true>(&mut got);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft84 inverse mismatch err={}", err);
}

#[test]
fn dft84_roundtrip() {
    let input = signal84();
    let mut data = input.clone();
    dft84_impl::<f64, false>(&mut data);
    dft84_impl::<f64, true>(&mut data);
    // Inverse is unnormalized, so we need to divide
    for x in &mut data {
        *x = *x / 84.0;
    }
    let err = max_abs_err_64(&data, &input);
    assert!(err < 1.0e-10, "dft84 roundtrip mismatch err={}", err);
}

#[test]
fn dft60_forward_matches_direct() {
    let input = signal60();
    let expected = dft_forward(&input);
    let mut got = input.clone();
    dft60_impl::<f64, false>(&mut got);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft60 forward mismatch err={}", err);
}

#[test]
fn dft60_inverse_matches_direct_unnormalized() {
    let input = signal60();
    let expected: Vec<_> = dft_inverse(&input).into_iter().map(|x| x * 60.0).collect();
    let mut got = input.clone();
    dft60_impl::<f64, true>(&mut got);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft60 inverse mismatch err={}", err);
}

#[test]
fn dft60_roundtrip() {
    let input = signal60();
    let mut data = input.clone();
    dft60_impl::<f64, false>(&mut data);
    dft60_impl::<f64, true>(&mut data);
    // Inverse is unnormalized, so we need to divide
    for x in &mut data {
        *x = *x / 60.0;
    }
    let err = max_abs_err_64(&data, &input);
    assert!(err < 1.0e-10, "dft60 roundtrip mismatch err={}", err);
}

#[test]
fn try_fft_dispatch_60() {
    let input = signal60();
    let expected = dft_forward(&input);
    let mut got = input.clone();
    // Dispatch with n1=4, n2=15
    let result = try_fft::<f64, false>(&mut got, 4, 15);
    assert!(result, "try_fft should return true for n1=4, n2=15");
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft60 dispatch forward mismatch err={}", err);
}

#[test]
fn try_fft_rejects_non_matching() {
    // Try to dispatch with n1=3, n2=20 (not Cook-Toom-GT sizes)
    let mut data = vec![num_complex::Complex::new(1.0, 0.0); 60];
    let result = try_fft::<f64, false>(&mut data, 3, 20);
    assert!(!result, "try_fft should return false for n1=3, n2=20");
}

fn signal90() -> Vec<Complex64> {
    (0..90)
        .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.29).cos()))
        .collect()
}

fn signal150() -> Vec<Complex64> {
    (0..150)
        .map(|k| Complex64::new((k as f64 * 0.13).sin(), (k as f64 * 0.41).cos()))
        .collect()
}

#[test]
fn dft90_forward_matches_direct() {
    let input = signal90();
    let expected = dft_forward(&input);
    let mut got = input.clone();
    dft90_impl::<f64, false>(&mut got);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft90 forward mismatch err={}", err);
}

#[test]
fn dft90_inverse_matches_direct_unnormalized() {
    let input = signal90();
    let expected: Vec<_> = dft_inverse(&input).into_iter().map(|x| x * 90.0).collect();
    let mut got = input.clone();
    dft90_impl::<f64, true>(&mut got);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft90 inverse mismatch err={}", err);
}

#[test]
fn dft90_roundtrip() {
    let input = signal90();
    let mut data = input.clone();
    dft90_impl::<f64, false>(&mut data);
    dft90_impl::<f64, true>(&mut data);
    for x in &mut data {
        *x = *x / 90.0;
    }
    let err = max_abs_err_64(&data, &input);
    assert!(err < 1.0e-10, "dft90 roundtrip mismatch err={}", err);
}

#[test]
fn dft150_forward_matches_direct() {
    let input = signal150();
    let expected = dft_forward(&input);
    let mut got = input.clone();
    dft150_impl::<f64, false>(&mut got);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft150 forward mismatch err={}", err);
}

#[test]
fn dft150_inverse_matches_direct_unnormalized() {
    let input = signal150();
    let expected: Vec<_> = dft_inverse(&input).into_iter().map(|x| x * 150.0).collect();
    let mut got = input.clone();
    dft150_impl::<f64, true>(&mut got);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft150 inverse mismatch err={}", err);
}

#[test]
fn dft150_roundtrip() {
    let input = signal150();
    let mut data = input.clone();
    dft150_impl::<f64, false>(&mut data);
    dft150_impl::<f64, true>(&mut data);
    for x in &mut data {
        *x = *x / 150.0;
    }
    let err = max_abs_err_64(&data, &input);
    assert!(err < 1.0e-10, "dft150 roundtrip mismatch err={}", err);
}

#[test]
fn try_fft_dispatch_90() {
    let input = signal90();
    let expected = dft_forward(&input);
    let mut got = input.clone();
    let result = try_fft::<f64, false>(&mut got, 9, 10);
    assert!(result, "try_fft should return true for n1=9, n2=10");
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "dft90 dispatch forward mismatch err={}", err);
}

#[test]
fn try_fft_dispatch_150() {
    let input = signal150();
    let expected = dft_forward(&input);
    let mut got = input.clone();
    let result = try_fft::<f64, false>(&mut got, 6, 25);
    assert!(result, "try_fft should return true for n1=6, n2=25");
    let err = max_abs_err_64(&got, &expected);
    assert!(
        err < 1.0e-10,
        "dft150 dispatch forward mismatch err={}",
        err
    );

    let expected_inv: Vec<_> = dft_inverse(&input).into_iter().map(|x| x * 150.0).collect();
    let mut got_inv = input.clone();
    let result_inv = try_fft::<f64, true>(&mut got_inv, 6, 25);
    assert!(
        result_inv,
        "try_fft inverse should return true for n1=6, n2=25"
    );
    let err_inv = max_abs_err_64(&got_inv, &expected_inv);
    assert!(
        err_inv < 1.0e-10,
        "dft150 dispatch inverse mismatch err={}",
        err_inv
    );
}
