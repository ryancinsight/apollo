use crate::application::execution::kernel::components::winograd::*;
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use eunomia::{Complex32, Complex64};

fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max)
}

// Ã¢â€â‚¬Ã¢â€â‚¬ DFT-16 Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

#[test]
fn dft16_forward_matches_direct() {
    let input: Vec<Complex64> = (0..16)
        .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.13).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
    dft16_impl::<f64, false>(&mut buf);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-11, "DFT-16 forward max_err={err:.2e}");
}

#[test]
fn dft16_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..16)
        .map(|k| Complex64::new((k as f64 * 0.06).cos(), (k as f64 * 0.19).sin()))
        .collect();
    let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
    dft16_impl::<f64, false>(&mut buf);
    dft16_impl::<f64, true>(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 16.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-11, "DFT-16 roundtrip max_err={err:.2e}");
}

#[test]
fn dft16_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..16)
        .map(|k| Complex64::new((k as f64 * 0.44).sin(), (k as f64 * 0.36).cos()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 16.0).collect();
    let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
    dft16_impl::<f64, true>(&mut buf);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-11, "DFT-16 inverse max_err={err:.2e}");
}

// Ã¢â€â‚¬Ã¢â€â‚¬ DFT-32 Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

#[test]
fn dft32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..32)
        .map(|k| Complex64::new((k as f64 * 0.21).sin(), (k as f64 * 0.09).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
    dft32_impl::<f64, false>(&mut buf);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-11, "DFT-32 forward max_err={err:.2e}");
}

#[test]
fn dft32_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..32)
        .map(|k| Complex64::new((k as f64 * 0.14).cos(), (k as f64 * 0.37).sin()))
        .collect();
    let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
    dft32_impl::<f64, false>(&mut buf);
    dft32_impl::<f64, true>(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 32.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-11, "DFT-32 roundtrip max_err={err:.2e}");
}

#[test]
fn dft32_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..32)
        .map(|k| Complex64::new((k as f64 * 0.55).sin(), (k as f64 * 0.27).cos()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 32.0).collect();
    let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
    dft32_impl::<f64, true>(&mut buf);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-11, "DFT-32 inverse max_err={err:.2e}");
}

#[test]
fn dft32_f32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..32)
        .map(|k| Complex64::new((k as f64 * 0.12).sin(), (k as f64 * 0.35).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 32] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    dft32_impl::<f32, false>(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 2e-5, "DFT-32 f32 forward max_err={err:.2e}");
}

// Ã¢â€â‚¬Ã¢â€â‚¬ DFT-64 Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

#[test]
fn dft64_forward_matches_direct() {
    let input: Vec<Complex64> = (0..64)
        .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.03).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
    dft64_impl::<f64, false>(&mut buf);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-11, "DFT-64 forward max_err={err:.2e}");
}

#[test]
fn dft64_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..64)
        .map(|k| Complex64::new((k as f64 * 0.08).cos(), (k as f64 * 0.51).sin()))
        .collect();
    let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
    dft64_impl::<f64, false>(&mut buf);
    dft64_impl::<f64, true>(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 64.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-11, "DFT-64 roundtrip max_err={err:.2e}");
}

#[test]
fn dft64_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..64)
        .map(|k| Complex64::new((k as f64 * 0.14).sin(), (k as f64 * 0.42).cos()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 64.0).collect();
    let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
    dft64_impl::<f64, true>(&mut buf);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-11, "DFT-64 inverse max_err={err:.2e}");
}

#[test]
fn dft64_f32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..64)
        .map(|k| Complex64::new((k as f64 * 0.07).sin(), (k as f64 * 0.29).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 64] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    dft64_impl::<f32, false>(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 3e-5, "DFT-64 f32 forward max_err={err:.2e}");
}
