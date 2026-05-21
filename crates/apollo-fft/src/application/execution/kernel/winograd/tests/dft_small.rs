use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::winograd::*;
use num_complex::{Complex32, Complex64};

fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max)
}

// ΓöÇΓöÇ DFT-3 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft3_forward_matches_direct() {
    let input: Vec<Complex64> = (0..3)
        .map(|k| Complex64::new((k as f64 * 0.71).sin(), (k as f64 * 0.43).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 3] = input.as_slice().try_into().unwrap();
    dft3_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-13, "DFT-3 forward max_err={err:.2e}");
}

#[test]
fn dft3_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..3)
        .map(|k| Complex64::new((k as f64 * 0.55).cos(), (k as f64 * 0.19).sin()))
        .collect();
    let mut buf: [Complex64; 3] = input.as_slice().try_into().unwrap();
    dft3_impl(&mut buf, false);
    dft3_impl(&mut buf, true);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 3.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-13, "DFT-3 roundtrip max_err={err:.2e}");
}

#[test]
fn dft3_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..3)
        .map(|k| Complex64::new((k as f64 * 0.39).cos(), (k as f64 * 0.83).sin()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 3.0).collect();
    let mut buf: [Complex64; 3] = input.as_slice().try_into().unwrap();
    dft3_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-13, "DFT-3 inverse max_err={err:.2e}");
}

#[test]
fn dft3_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 3];
    dft3_impl(&mut buf, false);
    assert!((buf[0] - Complex64::new(3.0, 0.0)).norm() < 1e-14);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-14, "non-zero bin: {:?}", x);
    }
}

// ΓöÇΓöÇ DFT-5 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft5_forward_matches_direct() {
    let input: Vec<Complex64> = (0..5)
        .map(|k| Complex64::new((k as f64 * 0.61).sin(), (k as f64 * 0.37).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 5] = input.as_slice().try_into().unwrap();
    dft5_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-12, "DFT-5 forward max_err={err:.2e}");
}

#[test]
fn dft5_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..5)
        .map(|k| Complex64::new((k as f64 * 0.47).cos(), (k as f64 * 0.28).sin()))
        .collect();
    let mut buf: [Complex64; 5] = input.as_slice().try_into().unwrap();
    dft5_impl(&mut buf, false);
    dft5_impl(&mut buf, true);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 5.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-12, "DFT-5 roundtrip max_err={err:.2e}");
}

#[test]
fn dft5_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..5)
        .map(|k| Complex64::new((k as f64 * 0.23).cos(), (k as f64 * 0.77).sin()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 5.0).collect();
    let mut buf: [Complex64; 5] = input.as_slice().try_into().unwrap();
    dft5_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-12, "DFT-5 inverse max_err={err:.2e}");
}

#[test]
fn dft5_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 5];
    dft5_impl(&mut buf, false);
    assert!((buf[0] - Complex64::new(5.0, 0.0)).norm() < 1e-14);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-14, "non-zero bin: {:?}", x);
    }
}

#[test]
fn dft5_f32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..5)
        .map(|k| Complex64::new((k as f64 * 0.53).sin(), (k as f64 * 0.31).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 5] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    dft5_impl(&mut buf, false);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 2e-6, "DFT-5 f32 forward max_err={err:.2e}");
}

// ΓöÇΓöÇ DFT-7 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft7_forward_matches_direct() {
    let input: Vec<Complex64> = (0..7)
        .map(|k| Complex64::new((k as f64 * 0.71).sin(), (k as f64 * 0.31).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 7] = input.as_slice().try_into().unwrap();
    dft7_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-12, "DFT-7 forward max_err={err:.2e}");
}

#[test]
fn dft7_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..7)
        .map(|k| Complex64::new((k as f64 * 0.37).cos(), (k as f64 * 0.19).sin()))
        .collect();
    let mut buf: [Complex64; 7] = input.as_slice().try_into().unwrap();
    dft7_impl(&mut buf, false);
    dft7_impl(&mut buf, true);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 7.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-12, "DFT-7 roundtrip max_err={err:.2e}");
}

#[test]
fn dft7_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..7)
        .map(|k| Complex64::new((k as f64 * 0.47).sin(), (k as f64 * 0.23).cos()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 7.0).collect();
    let mut buf: [Complex64; 7] = input.as_slice().try_into().unwrap();
    dft7_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-12, "DFT-7 inverse max_err={err:.2e}");
}

#[test]
fn dft7_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 7];
    dft7_impl(&mut buf, false);
    assert!((buf[0] - Complex64::new(7.0, 0.0)).norm() < 1e-14);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-14, "non-zero bin: {:?}", x);
    }
}

#[test]
fn dft7_f32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..7)
        .map(|k| Complex64::new((k as f64 * 0.53).sin(), (k as f64 * 0.31).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 7] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    dft7_impl(&mut buf, false);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 1e-5, "DFT-7 f32 forward max_err={err:.2e}");
}

// ΓöÇΓöÇ DFT-2 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft2_forward_matches_direct() {
    let input = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    let expected = dft_forward(&input);
    let mut a = input[0];
    let mut b = input[1];
    dft2_impl(&mut a, &mut b);
    assert!(
        max_err(&[a, b], &expected) < 1e-14,
        "DFT-2 forward mismatch"
    );
}

#[test]
fn dft2_inverse_roundtrip() {
    let mut a = Complex64::new(3.0, -1.0);
    let mut b = Complex64::new(-2.0, 4.0);
    let orig_a = a;
    let orig_b = b;
    // forward then unnorm-inverse should give 2├ù the original.
    dft2_impl(&mut a, &mut b);
    dft2_impl(&mut a, &mut b);
    assert!((a - 2.0 * orig_a).norm() < 1e-14);
    assert!((b - 2.0 * orig_b).norm() < 1e-14);
}

// ΓöÇΓöÇ DFT-4 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft4_forward_matches_direct() {
    let input: Vec<Complex64> = (0..4)
        .map(|k| Complex64::new((k as f64 * 0.3).sin(), (k as f64 * 0.7).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 4] = input.as_slice().try_into().unwrap();
    dft4_impl(&mut buf, false);
    assert!(max_err(&buf, &expected) < 1e-13, "DFT-4 forward mismatch");
}

#[test]
fn dft4_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..4)
        .map(|k| Complex64::new((k as f64 * 0.5).cos(), (k as f64 * 0.2).sin()))
        .collect();
    let mut buf: [Complex64; 4] = input.as_slice().try_into().unwrap();
    dft4_impl(&mut buf, false);
    dft4_impl(&mut buf, true);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 4.0).collect();
    assert!(
        max_err(&recovered, &input) < 1e-13,
        "DFT-4 roundtrip mismatch"
    );
}

#[test]
fn dft4_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..4)
        .map(|k| Complex64::new((k as f64 * 0.9).cos(), (k as f64 * 0.4).sin()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 4.0).collect();
    let mut buf: [Complex64; 4] = input.as_slice().try_into().unwrap();
    dft4_impl(&mut buf, true);
    assert!(
        max_err(&buf, &expected_unnorm) < 1e-13,
        "DFT-4 inverse mismatch"
    );
}

// ΓöÇΓöÇ DFT-8 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft8_forward_matches_direct() {
    let input: Vec<Complex64> = (0..8)
        .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.17).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 8] = input.as_slice().try_into().unwrap();
    dft8_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-12, "DFT-8 forward max_err={err:.2e}");
}

#[test]
fn dft8_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..8)
        .map(|k| Complex64::new((k as f64 * 0.23).cos(), -(k as f64 * 0.11).sin()))
        .collect();
    let mut buf: [Complex64; 8] = input.as_slice().try_into().unwrap();
    dft8_impl(&mut buf, false);
    dft8_impl(&mut buf, true);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 8.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-12, "DFT-8 roundtrip max_err={err:.2e}");
}

#[test]
fn dft8_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..8)
        .map(|k| Complex64::new((k as f64 * 0.33).sin(), (k as f64 * 0.22).cos()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 8.0).collect();
    let mut buf: [Complex64; 8] = input.as_slice().try_into().unwrap();
    dft8_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-12, "DFT-8 inverse max_err={err:.2e}");
}

#[test]
fn dft8_f32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..8)
        .map(|k| Complex64::new((k as f64 * 0.18).sin(), (k as f64 * 0.31).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 8] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    dft8_impl(&mut buf, false);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 1e-5, "DFT-8 f32 forward max_err={err:.2e}");
}

// ΓöÇΓöÇ DFT-16 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft16_forward_matches_direct() {
    let input: Vec<Complex64> = (0..16)
        .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.13).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
    dft16_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-11, "DFT-16 forward max_err={err:.2e}");
}

#[test]
fn dft16_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..16)
        .map(|k| Complex64::new((k as f64 * 0.06).cos(), (k as f64 * 0.19).sin()))
        .collect();
    let mut buf: [Complex64; 16] = input.as_slice().try_into().unwrap();
    dft16_impl(&mut buf, false);
    dft16_impl(&mut buf, true);
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
    dft16_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-11, "DFT-16 inverse max_err={err:.2e}");
}

// ΓöÇΓöÇ DFT-32 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..32)
        .map(|k| Complex64::new((k as f64 * 0.21).sin(), (k as f64 * 0.09).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
    dft32_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-11, "DFT-32 forward max_err={err:.2e}");
}

#[test]
fn dft32_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..32)
        .map(|k| Complex64::new((k as f64 * 0.14).cos(), (k as f64 * 0.37).sin()))
        .collect();
    let mut buf: [Complex64; 32] = input.as_slice().try_into().unwrap();
    dft32_impl(&mut buf, false);
    dft32_impl(&mut buf, true);
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
    dft32_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-11, "DFT-32 inverse max_err={err:.2e}");
}

// ΓöÇΓöÇ DFT-64 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft64_forward_matches_direct() {
    let input: Vec<Complex64> = (0..64)
        .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.03).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
    dft64_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-11, "DFT-64 forward max_err={err:.2e}");
}

#[test]
fn dft64_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..64)
        .map(|k| Complex64::new((k as f64 * 0.08).cos(), (k as f64 * 0.51).sin()))
        .collect();
    let mut buf: [Complex64; 64] = input.as_slice().try_into().unwrap();
    dft64_impl(&mut buf, false);
    dft64_impl(&mut buf, true);
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
    dft64_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-11, "DFT-64 inverse max_err={err:.2e}");
}

// ΓöÇΓöÇ boundary cases ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

#[test]
fn dft4_impulse_produces_all_ones() {
    // DFT([1,0,0,0]) = [1,1,1,1]  (Cooley & Tukey 1965)
    let mut buf = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    dft4_impl(&mut buf, false);
    for x in &buf {
        assert!((x - Complex64::new(1.0, 0.0)).norm() < 1e-14);
    }
}

#[test]
fn dft8_dc_produces_energy_in_bin0() {
    // DFT([1,1,1,1,1,1,1,1]) = [8,0,0,0,0,0,0,0]
    let mut buf = [Complex64::new(1.0, 0.0); 8];
    dft8_impl(&mut buf, false);
    assert!((buf[0] - Complex64::new(8.0, 0.0)).norm() < 1e-12);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-12);
    }
}

#[test]
fn dft16_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 16];
    dft16_impl(&mut buf, false);
    assert!((buf[0] - Complex64::new(16.0, 0.0)).norm() < 1e-11);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-11);
    }
}

#[test]
fn dft32_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 32];
    dft32_impl(&mut buf, false);
    assert!((buf[0] - Complex64::new(32.0, 0.0)).norm() < 1e-11);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-11);
    }
}

#[test]
fn dft64_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 64];
    dft64_impl(&mut buf, false);
    assert!((buf[0] - Complex64::new(64.0, 0.0)).norm() < 1e-11);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-11);
    }
}

#[test]
fn dft32_f32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..32)
        .map(|k| Complex64::new((k as f64 * 0.12).sin(), (k as f64 * 0.35).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 32] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    dft32_impl(&mut buf, false);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 2e-5, "DFT-32 f32 forward max_err={err:.2e}");
}

#[test]
fn dft64_f32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..64)
        .map(|k| Complex64::new((k as f64 * 0.07).sin(), (k as f64 * 0.29).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 64] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    dft64_impl(&mut buf, false);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 3e-5, "DFT-64 f32 forward max_err={err:.2e}");
}
