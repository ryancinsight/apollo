//! Test suite for the radix-composite kernel.
//!
//! Submodules group tests by concern:
//! - [`factorize`]: `factorize_composite` support/rejection contract
//! - [`correctness`]: forward/roundtrip/inverse value-semantics for canonical sizes
//! - [`prime_radix`]: extended prime-radix coverage (11, 13)
//! - [`large_prime`]: 17 and 23 prime-radix coverage
//! - [`composite`]: three-odd-prime + radix-4 tail composites

use super::*;
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::radix_shape::factorize_composite;
use eunomia::{Complex32, Complex64};

mod composite;
mod correctness;
mod factorize;
mod large_prime;
mod prime_radix;

pub(super) fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max)
}

pub(super) fn forward_inplace_64(data: &mut [Complex64]) {
    let radices = factorize_composite(data.len()).expect("test length must be prime23-smooth");
    forward_inplace_with_radices(data, &radices);
}

pub(super) fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    let radices = factorize_composite(data.len()).expect("test length must be prime23-smooth");
    inverse_inplace_unnorm_with_radices(data, &radices);
}

pub(super) fn forward_inplace_32(data: &mut [Complex32]) {
    let radices = factorize_composite(data.len()).expect("test length must be prime23-smooth");
    forward_inplace_with_radices(data, &radices);
}

pub(super) fn check_forward(n: usize, tol: f64) {
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

pub(super) fn check_roundtrip(n: usize, tol: f64) {
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.53).cos(), (k as f64 * 0.27).sin()))
        .collect();
    let mut buf = input.clone();
    forward_inplace_64(&mut buf);
    inverse_inplace_unnorm_64(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| *x / n as f64).collect();
    let err = max_err(&recovered, &input);
    assert!(
        err < tol,
        "roundtrip N={n}: max_err={err:.2e} (tol={tol:.2e})"
    );
}

pub(super) fn check_inverse(n: usize, tol: f64) {
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
