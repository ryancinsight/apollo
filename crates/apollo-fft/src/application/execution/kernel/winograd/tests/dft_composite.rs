use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::winograd::*;
use num_complex::{Complex32, Complex64};

fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max)
}

// DFT-15

#[test]
fn dft15_forward_matches_direct() {
    let input: Vec<Complex64> = (0..15)
        .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.27).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 15] = input.as_slice().try_into().unwrap();
    dft15_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-12, "DFT-15 forward max_err={err:.2e}");
}

#[test]
fn dft15_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..15)
        .map(|k| Complex64::new((k as f64 * 0.33).cos(), (k as f64 * 0.61).sin()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 15.0).collect();
    let mut buf: [Complex64; 15] = input.as_slice().try_into().unwrap();
    dft15_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-12, "DFT-15 inverse max_err={err:.2e}");
}

#[test]
fn dft15_roundtrip_recovers_input() {
    let input: Vec<Complex64> = (0..15)
        .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.53).cos()))
        .collect();
    let mut buf: [Complex64; 15] = input.as_slice().try_into().unwrap();
    dft15_impl(&mut buf, false);
    dft15_impl(&mut buf, true);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 15.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-12, "DFT-15 roundtrip max_err={err:.2e}");
}

#[test]
fn dft15_dc_energy_in_bin0_only() {
    let mut buf = [Complex64::new(1.0, 0.0); 15];
    dft15_impl(&mut buf, false);
    assert!(
        (buf[0] - Complex64::new(15.0, 0.0)).norm() < 1e-12,
        "DC bin: {:?}",
        buf[0]
    );
    for (k, x) in buf[1..].iter().enumerate() {
        assert!(x.norm() < 1e-12, "non-zero bin[{}]: {:?}", k + 1, x);
    }
}

#[test]
fn dft15_f32_forward_matches_f64() {
    let input64: Vec<Complex64> = (0..15)
        .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.27).cos()))
        .collect();
    let input32: Vec<Complex32> = input64
        .iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect();
    let mut buf64: [Complex64; 15] = input64.as_slice().try_into().unwrap();
    let mut buf32: [Complex32; 15] = input32.as_slice().try_into().unwrap();
    dft15_impl(&mut buf64, false);
    dft15_impl(&mut buf32, false);
    let err = buf32
        .iter()
        .zip(buf64.iter())
        .map(|(a, b)| {
            let diff = Complex32::new((a.re - b.re as f32).abs(), (a.im - b.im as f32).abs());
            diff.re.max(diff.im)
        })
        .fold(0.0f32, f32::max);
    assert!(err < 1e-4, "f32 vs f64 DFT-15 max_err={err:.2e}");
}
