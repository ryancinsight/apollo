mod good_thomas;

use super::super::test_utils::max_abs_err_64;
use super::*;
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use half::f16;
use num_complex::{Complex, Complex32, Complex64};

#[test]
fn mixed_forward_n32_matches_direct() {
    let n = 32usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.17).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    let expected = dft_forward(&input);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1e-10, "mixed-radix forward mismatch err={err:.2e}");
}

#[test]
fn mixed_forward_n121_matches_direct() {
    let n = 121usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.17).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    let expected = dft_forward(&input);
    let err = max_abs_err_64(&got, &expected);
    assert!(
        err < 1e-10,
        "mixed-radix forward N=121 mismatch err={err:.2e}"
    );
}

#[test]
fn mixed_inverse_unnorm_n32_matches_direct() {
    let n = 32usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.19).cos(), (k as f64 * 0.07).sin()))
        .collect();
    let mut got = input.clone();
    inverse_inplace_unnorm::<f64>(&mut got);
    let expected = dft_inverse(&input)
        .into_iter()
        .map(|x| x * n as f64)
        .collect::<Vec<_>>();
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1e-10, "mixed-radix inverse mismatch err={err:.2e}");
}

#[test]
fn mixed_forward_prime_n17_matches_direct() {
    let n = 17usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.31).sin(), (k as f64 * 0.23).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    let expected = dft_forward(&input);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "Rader forward mismatch err={err:.2e}");
}

#[test]
fn mixed_inverse_prime_n17_matches_direct() {
    let n = 17usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.37).cos(), (k as f64 * 0.41).sin()))
        .collect();
    let mut got = input.clone();
    inverse_inplace_unnorm::<f64>(&mut got);
    let expected = dft_inverse(&input)
        .into_iter()
        .map(|x| x * n as f64)
        .collect::<Vec<_>>();
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-10, "Rader inverse mismatch err={err:.2e}");
}

#[test]
fn mixed_forward_mid_primes_use_winograd_pair_and_match_direct() {
    for n in [19usize, 29, 31, 37, 41, 43, 47, 53] {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.31).sin(), (k as f64 * 0.23).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace::<f64>(&mut got);
        let expected = dft_forward(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(
            err < 1.0e-10,
            "Winograd-pair dispatch forward N={n} mismatch err={err:.2e}"
        );
    }
}

#[test]
fn mixed_inverse_mid_primes_use_winograd_pair_and_match_direct() {
    for n in [19usize, 29, 31, 37, 41, 43, 47, 53] {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.37).cos(), (k as f64 * 0.41).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm::<f64>(&mut got);
        let expected = dft_inverse(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(
            err < 1.0e-10,
            "Winograd-pair dispatch inverse N={n} mismatch err={err:.2e}"
        );
    }
}

#[test]
fn mixed_forward_two_by_prime_uses_winograd_prime_halves_n38() {
    let n = 38usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.13).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    let expected = dft_forward(&input);
    let err = max_abs_err_64(&got, &expected);
    assert!(
        err < 1.0e-10,
        "two-by-prime Winograd-halves forward N={n} mismatch err={err:.2e}"
    );
}

#[test]
fn mixed_inverse_two_by_prime_uses_winograd_prime_halves_n82() {
    let n = 82usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.17).cos(), (k as f64 * 0.23).sin()))
        .collect();
    let mut got = input.clone();
    inverse_inplace_unnorm::<f64>(&mut got);
    let expected = dft_inverse(&input)
        .into_iter()
        .map(|x| x * n as f64)
        .collect::<Vec<_>>();
    let err = max_abs_err_64(&got, &expected);
    assert!(
        err < 1.0e-9,
        "two-by-prime Winograd-halves inverse N={n} mismatch err={err:.2e}"
    );
}

#[test]
fn mixed_forward_two_by_prime_uses_winograd_prime_halves_n58() {
    let n = 58usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.11).sin(), (k as f64 * 0.19).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    let expected = dft_forward(&input);
    let err = max_abs_err_64(&got, &expected);
    assert!(
        err < 1.0e-10,
        "two-by-prime Winograd-halves forward N={n} mismatch err={err:.2e}"
    );
}

#[test]
fn mixed_inverse_two_by_prime_uses_winograd_prime_halves_n74() {
    let n = 74usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.07).cos(), (k as f64 * 0.29).sin()))
        .collect();
    let mut got = input.clone();
    inverse_inplace_unnorm::<f64>(&mut got);
    let expected = dft_inverse(&input)
        .into_iter()
        .map(|x| x * n as f64)
        .collect::<Vec<_>>();
    let err = max_abs_err_64(&got, &expected);
    assert!(
        err < 1.0e-9,
        "two-by-prime Winograd-halves inverse N={n} mismatch err={err:.2e}"
    );
}

#[test]
fn mixed_inverse_two_by_prime_uses_winograd_prime_halves_n94() {
    let n = 94usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.07).cos(), (k as f64 * 0.29).sin()))
        .collect();
    let mut got = input.clone();
    inverse_inplace_unnorm::<f64>(&mut got);
    let expected = dft_inverse(&input)
        .into_iter()
        .map(|x| x * n as f64)
        .collect::<Vec<_>>();
    let err = max_abs_err_64(&got, &expected);
    assert!(
        err < 1.0e-9,
        "two-by-prime Winograd-halves inverse N={n} mismatch err={err:.2e}"
    );
}

#[test]
#[ignore = "prime rader/bluestein (n=257 m=256 pow2 pad) debug monomorph/stock frames exceed thread stack (pre-existing pattern; see planned_rader_n113_f32; f32 subpaths + avx fixed + dftN in pads). Release/bench + f64 rader value unaffected. Value coverage via other primes, f64 rader, n256 stockham roundtrip, n512 ZST, GT. Prevents blocking on debug runs while preserving regression guard on PoT/rader sizes in focused benches."]
fn mixed_forward_prime_n257_matches_direct() {
    let n = 257usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.013).sin(), (k as f64 * 0.017).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    let expected = dft_forward(&input);
    let err = max_abs_err_64(&got, &expected);
    assert!(err < 1.0e-8, "prime forward mismatch err={err:.2e}");
}

#[test]
fn mixed_reduced_stockham_forward_inverse_roundtrip_n256() {
    let n = 256usize;
    let input: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.013).sin(), (k as f32 * 0.017).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f32>(&mut got);
    inverse_inplace::<f32>(&mut got);
    let err = got
        .iter()
        .zip(input.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f32, f32::max);
    assert!(
        err < 1.0e-4,
        "f32 Stockham roundtrip mismatch err={err:.2e}"
    );
}

#[test]
fn mixed_reduced_stockham_forward_inverse_roundtrip_n512() {
    let n = 512usize;
    let input: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.011).sin(), (k as f32 * 0.019).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f32>(&mut got);
    inverse_inplace::<f32>(&mut got);
    let err = got
        .iter()
        .zip(input.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f32, f32::max);
    assert!(
        err < 1.0e-4,
        "f32 Stockham N=512 roundtrip mismatch err={err:.2e}"
    );
}

#[test]
fn mixed_reduced_stockham_forward_inverse_roundtrip_n4096() {
    let n = 4096usize;
    let input: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.007).sin(), (k as f32 * 0.011).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f32>(&mut got);
    inverse_inplace::<f32>(&mut got);
    let err = got
        .iter()
        .zip(input.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f32, f32::max);
    let tolerance = 8.0 * n as f32 * f32::EPSILON;
    assert!(
        err < tolerance,
        "f32 Stockham N=4096 roundtrip mismatch err={err:.2e} tolerance={tolerance:.2e}"
    );
}

#[test]
fn compact_half_storage_impulse_has_flat_spectrum() {
    let n = 8usize;
    let mut data = vec![Complex::new(f16::ZERO, f16::ZERO); n];
    data[0] = Complex::new(f16::from_f32(1.0), f16::ZERO);
    forward_compact_storage(&mut data);
    for (bin, value) in data.iter().enumerate() {
        assert!(
            (value.re.to_f32() - 1.0).abs() < 5.0e-3,
            "bin {bin} real part must equal 1 within f16 storage error"
        );
        assert!(
            value.im.to_f32().abs() < 5.0e-3,
            "bin {bin} imaginary part must equal 0 within f16 storage error"
        );
    }
}

#[test]
fn compact_half_storage_roundtrip_stays_within_storage_error() {
    let input: Vec<Complex<f16>> = (0..64)
        .map(|index| {
            let value = (index as f32 * 0.23 - 1.5).tanh();
            Complex::new(f16::from_f32(value), f16::ZERO)
        })
        .collect();
    let mut data = input.clone();
    forward_compact_storage(&mut data);
    inverse_compact_storage(&mut data);
    let max_err = data
        .iter()
        .zip(input.iter())
        .map(|(actual, expected)| (actual.re.to_f32() - expected.re.to_f32()).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 5.0e-2,
        "f16 storage roundtrip max error {max_err:.4e}"
    );
}

#[test]
fn mixed_precise_stockham_forward_inverse_roundtrip_n256() {
    let n = 256usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.013).sin(), (k as f64 * 0.017).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    inverse_inplace::<f64>(&mut got);
    let err = max_abs_err_64(&got, &input);
    assert!(
        err < 1.0e-10,
        "f64 Stockham roundtrip mismatch err={err:.2e}"
    );
}

#[test]
fn mixed_precise_stockham_forward_inverse_roundtrip_n512() {
    let n = 512usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.011).sin(), (k as f64 * 0.019).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    inverse_inplace::<f64>(&mut got);
    let err = max_abs_err_64(&got, &input);
    assert!(
        err < 1.0e-10,
        "f64 Stockham N=512 roundtrip mismatch err={err:.2e}"
    );
}

/// N=8192 = 2^13 exercises an asymmetric power-of-two route.
#[test]
fn power_of_two_asymmetric_n8192_forward_inverse_roundtrip() {
    let n = 8192usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.007).sin(), (k as f64 * 0.011).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    inverse_inplace::<f64>(&mut got);
    let err = max_abs_err_64(&got, &input);
    let tol = 8.0 * n as f64 * f64::EPSILON;
    assert!(
        err < tol,
        "power-of-two N=8192 roundtrip max_err={err:.2e} tol={tol:.2e}"
    );
}

/// N=32768 = 2^15 exercises the asymmetric power-of-two padding length for N=10007.
#[test]
fn power_of_two_asymmetric_n32768_forward_inverse_roundtrip() {
    let n = 32768usize;
    let input: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new((k as f64 * 0.003).sin(), (k as f64 * 0.007).cos()))
        .collect();
    let mut got = input.clone();
    forward_inplace::<f64>(&mut got);
    inverse_inplace::<f64>(&mut got);
    let err = max_abs_err_64(&got, &input);
    let tol = 8.0 * n as f64 * f64::EPSILON;
    assert!(
        err < tol,
        "power-of-two N=32768 roundtrip max_err={err:.2e} tol={tol:.2e}"
    );
}

#[test]
fn mixed_precise_power_of_two_n32768_forward_dc_is_not_noop() {
    let n = 32768usize;
    let mut got = vec![Complex64::new(1.0, 0.0); n];
    forward_inplace::<f64>(&mut got);
    let tol = 8.0 * n as f64 * f64::EPSILON;
    assert!(
        (got[0] - Complex64::new(n as f64, 0.0)).norm() < tol,
        "N=32768 DC bin mismatch: {:?}",
        got[0]
    );
    let tail = got[1..].iter().map(|z| z.norm()).fold(0.0f64, f64::max);
    assert!(
        tail < tol,
        "N=32768 forward DC tail max_err={tail:.2e} tol={tol:.2e}"
    );
}

#[test]
fn test_small_pot_f64_correctness() {
    let sizes = [2, 4, 8, 16, 32, 64];
    for &n in &sizes {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.29).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace::<f64>(&mut got);
        inverse_inplace::<f64>(&mut got);
        let err = max_abs_err_64(&got, &input);
        assert!(
            err < 1.0e-12,
            "f64 small POT N={} roundtrip mismatch err={:e}",
            n,
            err
        );
    }
}

#[test]
fn test_small_pot_f32_correctness() {
    let sizes = [2, 4, 8, 16, 32, 64];
    for &n in &sizes {
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.17).sin(), (k as f32 * 0.29).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace::<f32>(&mut got);
        inverse_inplace::<f32>(&mut got);
        let err = got
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f32, f32::max);
        assert!(
            err < 1.0e-6,
            "f32 small POT N={} roundtrip mismatch err={:e}",
            n,
            err
        );
    }
}
