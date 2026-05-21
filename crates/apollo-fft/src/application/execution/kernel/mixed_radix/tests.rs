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
fn mixed_f32_stockham_forward_inverse_roundtrip_n256() {
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
fn mixed_f32_stockham_forward_inverse_roundtrip_n512() {
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
fn mixed_f32_stockham_forward_inverse_roundtrip_n4096() {
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
fn compact_f16_storage_impulse_has_flat_spectrum() {
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
fn compact_f16_storage_roundtrip_stays_within_storage_error() {
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
fn mixed_f64_stockham_forward_inverse_roundtrip_n256() {
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
fn mixed_f64_stockham_forward_inverse_roundtrip_n512() {
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
