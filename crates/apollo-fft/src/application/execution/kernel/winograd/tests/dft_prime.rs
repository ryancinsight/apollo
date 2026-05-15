use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::winograd::*;
use num_complex::{Complex32, Complex64};

fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max)
}

#[test]
fn dft17_forward_matches_direct() {
    let input: Vec<Complex64> = (0..17)
        .map(|k| Complex64::new((k as f64 * 0.19).sin(), (k as f64 * 0.37).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 17] = input.as_slice().try_into().unwrap();
    dft17_impl::<f64, false>(&mut buf);
    let err = max_err(&buf, &expected);
    assert!(err < 2e-12, "DFT-17 forward max_err={err:.2e}");
}

#[test]
fn dft17_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..17)
        .map(|k| Complex64::new((k as f64 * 0.29).cos(), -(k as f64 * 0.13).sin()))
        .collect();
    let mut buf: [Complex64; 17] = input.as_slice().try_into().unwrap();
    dft17_impl::<f64, false>(&mut buf);
    dft17_impl::<f64, true>(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 17.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 2e-12, "DFT-17 roundtrip max_err={err:.2e}");
}

#[test]
fn dft17_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..17)
        .map(|k| Complex64::new((k as f64 * 0.43).sin(), (k as f64 * 0.31).cos()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 17.0).collect();
    let mut buf: [Complex64; 17] = input.as_slice().try_into().unwrap();
    dft17_impl::<f64, true>(&mut buf);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 2e-12, "DFT-17 inverse max_err={err:.2e}");
}

#[test]
fn dft17_f32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..17)
        .map(|k| Complex64::new((k as f64 * 0.23).sin(), (k as f64 * 0.41).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 17] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    dft17_impl::<f32, false>(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 2e-5, "DFT-17 f32 forward max_err={err:.2e}");
}

#[test]
fn dft23_forward_matches_direct() {
    let input: Vec<Complex64> = (0..23)
        .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.35).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 23] = input.as_slice().try_into().unwrap();
    dft23_impl::<f64, false>(&mut buf);
    let err = max_err(&buf, &expected);
    assert!(err < 4e-12, "DFT-23 forward max_err={err:.2e}");
}

#[test]
fn dft23_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..23)
        .map(|k| Complex64::new((k as f64 * 0.27).cos(), -(k as f64 * 0.11).sin()))
        .collect();
    let mut buf: [Complex64; 23] = input.as_slice().try_into().unwrap();
    dft23_impl::<f64, false>(&mut buf);
    dft23_impl::<f64, true>(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 23.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 4e-12, "DFT-23 roundtrip max_err={err:.2e}");
}

#[test]
fn dft23_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..23)
        .map(|k| Complex64::new((k as f64 * 0.39).sin(), (k as f64 * 0.29).cos()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 23.0).collect();
    let mut buf: [Complex64; 23] = input.as_slice().try_into().unwrap();
    dft23_impl::<f64, true>(&mut buf);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 4e-12, "DFT-23 inverse max_err={err:.2e}");
}

#[test]
fn dft23_f32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..23)
        .map(|k| Complex64::new((k as f64 * 0.21).sin(), (k as f64 * 0.43).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 23] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    dft23_impl::<f32, false>(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 4e-5, "DFT-23 f32 forward max_err={err:.2e}");
}
