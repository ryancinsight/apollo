use crate::application::execution::kernel::components::winograd::*;
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::mixed_radix::traits::ShortDft;
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
    let recovered: Vec<Complex64> = buf.iter().map(|x| *x / 16.0).collect();
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
    let recovered: Vec<Complex64> = buf.iter().map(|x| *x / 32.0).collect();
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
fn routed_dft32_forward_matches_direct() {
    let input: Vec<Complex64> = (0..32)
        .map(|k| Complex64::new((k as f64 * 0.12).sin(), (k as f64 * 0.35).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex32; 32] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    <f32 as ShortDft<32>>::dft::<false>(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    let input_l1 = input.iter().map(|value| value.norm()).sum::<f64>();
    let bound = 64.0 * f64::from(f32::EPSILON) * input_l1.max(1.0);
    assert!(
        err <= bound,
        "routed DFT-32 forward max_err={err:.2e}, bound={bound:.2e}"
    );
}

#[test]
fn routed_dft32_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..32)
        .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.23).cos()))
        .collect();
    let expected: Vec<Complex64> = dft_inverse(&input)
        .into_iter()
        .map(|value| value * 32.0)
        .collect();
    let mut buf = core::array::from_fn(|index| {
        Complex32::new(input[index].re as f32, input[index].im as f32)
    });
    <f32 as ShortDft<32>>::dft::<true>(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    let input_l1 = input.iter().map(|value| value.norm()).sum::<f64>();
    let bound = 64.0 * f64::from(f32::EPSILON) * input_l1.max(1.0);
    assert!(
        err <= bound,
        "routed DFT-32 inverse max_err={err:.2e}, bound={bound:.2e}"
    );
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
    let recovered: Vec<Complex64> = buf.iter().map(|x| *x / 64.0).collect();
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

#[test]
fn dft96_forward_inverse_match_direct_for_each_scalar() {
    let input64: [Complex64; 96] = core::array::from_fn(|index| {
        let x = index as f64;
        Complex64::new((0.17 * x).sin(), 0.25 * (0.31 * x).cos())
    });
    let input_l1_64 = input64.iter().map(|value| value.norm()).sum::<f64>();
    // Each direct/codelet comparison combines two two-level transforms. Four
    // rounded operations per input term is a conservative first-order bound.
    let bound64 = 4.0 * 96.0 * f64::EPSILON * input_l1_64.max(1.0);
    let expected_forward64 = dft_forward(&input64);
    let mut forward64 = input64;
    <f64 as ShortDft<96>>::dft::<false>(&mut forward64);
    let forward_error64 = max_err(&forward64, &expected_forward64);
    assert!(
        forward_error64 <= bound64,
        "DFT-96 f64 forward error={forward_error64:.3e}, bound={bound64:.3e}"
    );

    let expected_inverse64 = dft_inverse(&input64)
        .into_iter()
        .map(|value| value * 96.0)
        .collect::<Vec<_>>();
    let mut inverse64 = input64;
    <f64 as ShortDft<96>>::dft::<true>(&mut inverse64);
    let inverse_error64 = max_err(&inverse64, &expected_inverse64);
    assert!(
        inverse_error64 <= bound64,
        "DFT-96 f64 inverse error={inverse_error64:.3e}, bound={bound64:.3e}"
    );

    let input32: [Complex32; 96] = core::array::from_fn(|index| {
        let x = index as f32;
        Complex32::new((0.17 * x).sin(), 0.25 * (0.31 * x).cos())
    });
    let input_l1_32 = input32
        .iter()
        .map(|value| f64::from(value.norm()))
        .sum::<f64>();
    let bound32 = 4.0 * 96.0 * f64::from(f32::EPSILON) * input_l1_32.max(1.0);
    let expected_forward32 = dft_forward(&input32);
    let mut forward32 = input32;
    <f32 as ShortDft<96>>::dft::<false>(&mut forward32);
    let forward_error32 = forward32
        .iter()
        .zip(&expected_forward32)
        .map(|(actual, expected)| f64::from((*actual - *expected).norm()))
        .fold(0.0_f64, f64::max);
    assert!(
        forward_error32 <= bound32,
        "DFT-96 f32 forward error={forward_error32:.3e}, bound={bound32:.3e}"
    );

    let expected_inverse32 = dft_inverse(&input32)
        .into_iter()
        .map(|value| value * 96.0_f32)
        .collect::<Vec<_>>();
    let mut inverse32 = input32;
    <f32 as ShortDft<96>>::dft::<true>(&mut inverse32);
    let inverse_error32 = inverse32
        .iter()
        .zip(&expected_inverse32)
        .map(|(actual, expected)| f64::from((*actual - *expected).norm()))
        .fold(0.0_f64, f64::max);
    assert!(
        inverse_error32 <= bound32,
        "DFT-96 f32 inverse error={inverse_error32:.3e}, bound={bound32:.3e}"
    );
}
