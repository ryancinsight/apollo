use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::winograd::*;
use num_complex::{Complex32, Complex64};

fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max)
}

fn roundoff_bound(input: &[Complex64], rounded_real_ops: usize) -> f64 {
    let t = rounded_real_ops as f64 * f64::EPSILON;
    let gamma = t / (1.0 - t);
    gamma * input.iter().map(|z| z.norm()).sum::<f64>()
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

// DFT-25

#[test]
fn dft25_forward_matches_direct() {
    let input: Vec<Complex64> = (0..25)
        .map(|k| Complex64::new((k as f64 * 0.37).sin(), (k as f64 * 0.19).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 25] = input.as_slice().try_into().unwrap();
    dft25_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    assert!(err < 1e-9, "DFT-25 forward max_err={err:.2e}");
}

#[test]
fn dft25_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..25)
        .map(|k| Complex64::new((k as f64 * 0.43).cos(), (k as f64 * 0.71).sin()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 25.0).collect();
    let mut buf: [Complex64; 25] = input.as_slice().try_into().unwrap();
    dft25_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-9, "DFT-25 inverse max_err={err:.2e}");
}

#[test]
fn dft25_roundtrip_recovers_input() {
    let input: Vec<Complex64> = (0..25)
        .map(|k| Complex64::new((k as f64 * 0.23).sin(), (k as f64 * 0.47).cos()))
        .collect();
    let mut buf: [Complex64; 25] = input.as_slice().try_into().unwrap();
    dft25_impl(&mut buf, false);
    dft25_impl(&mut buf, true);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 25.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-9, "DFT-25 roundtrip max_err={err:.2e}");
}

#[test]
fn dft25_dc_energy_in_bin0_only() {
    let mut buf = [Complex64::new(1.0, 0.0); 25];
    dft25_impl(&mut buf, false);
    assert!(
        (buf[0] - Complex64::new(25.0, 0.0)).norm() < 1e-11,
        "DC bin: {:?}",
        buf[0]
    );
    for (k, x) in buf[1..].iter().enumerate() {
        assert!(x.norm() < 1e-11, "non-zero bin[{}]: {:?}", k + 1, x);
    }
}

#[test]
fn dft25_f32_forward_matches_f64() {
    let input64: Vec<Complex64> = (0..25)
        .map(|k| Complex64::new((k as f64 * 0.37).sin(), (k as f64 * 0.19).cos()))
        .collect();
    let input32: Vec<Complex32> = input64
        .iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect();
    let mut buf64: [Complex64; 25] = input64.as_slice().try_into().unwrap();
    let mut buf32: [Complex32; 25] = input32.as_slice().try_into().unwrap();
    dft25_impl(&mut buf64, false);
    dft25_impl(&mut buf32, false);
    let err = buf32
        .iter()
        .zip(buf64.iter())
        .map(|(a, b)| {
            let diff = Complex32::new((a.re - b.re as f32).abs(), (a.im - b.im as f32).abs());
            diff.re.max(diff.im)
        })
        .fold(0.0f32, f32::max);
    assert!(err < 1e-4, "f32 vs f64 DFT-25 max_err={err:.2e}");
}

// DFT-100

#[test]
fn dft100_forward_matches_direct() {
    let input: Vec<Complex64> = (0..100)
        .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.17).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 100] = input.as_slice().try_into().unwrap();
    dft100_impl(&mut buf, false);
    let err = max_err(&buf, &expected);
    let bound = roundoff_bound(&input, 10_000);
    assert!(
        err <= bound,
        "DFT-100 forward max_err={err:.2e}, bound={bound:.2e}"
    );
}

#[test]
fn dft100_inverse_matches_direct() {
    let input: Vec<Complex64> = (0..100)
        .map(|k| Complex64::new((k as f64 * 0.31).cos(), (k as f64 * 0.23).sin()))
        .collect();
    let expected_unnorm: Vec<Complex64> =
        dft_inverse(&input).into_iter().map(|x| x * 100.0).collect();
    let mut buf: [Complex64; 100] = input.as_slice().try_into().unwrap();
    dft100_impl(&mut buf, true);
    let err = max_err(&buf, &expected_unnorm);
    let bound = roundoff_bound(&input, 10_000);
    assert!(
        err <= bound,
        "DFT-100 inverse max_err={err:.2e}, bound={bound:.2e}"
    );
}

#[test]
fn dft100_roundtrip_recovers_input() {
    let input: Vec<Complex64> = (0..100)
        .map(|k| Complex64::new((k as f64 * 0.13).sin(), (k as f64 * 0.41).cos()))
        .collect();
    let mut buf: [Complex64; 100] = input.as_slice().try_into().unwrap();
    dft100_impl(&mut buf, false);
    dft100_impl(&mut buf, true);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 100.0).collect();
    let err = max_err(&recovered, &input);
    let bound = roundoff_bound(&input, 20_000);
    assert!(
        err <= bound,
        "DFT-100 roundtrip max_err={err:.2e}, bound={bound:.2e}"
    );
}

#[test]
fn dft100_dc_energy_in_bin0_only() {
    let mut buf = [Complex64::new(1.0, 0.0); 100];
    dft100_impl(&mut buf, false);
    let bound = roundoff_bound(&buf, 10_000);
    assert!(
        (buf[0] - Complex64::new(100.0, 0.0)).norm() <= bound,
        "DC bin: {:?}, bound={bound:.2e}",
        buf[0]
    );
    for (k, x) in buf[1..].iter().enumerate() {
        assert!(
            x.norm() <= bound,
            "non-zero bin[{}]: {:?}, bound={bound:.2e}",
            k + 1,
            x
        );
    }
}

#[test]
fn dft100_f32_forward_matches_f64() {
    let input64: Vec<Complex64> = (0..100)
        .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.17).cos()))
        .collect();
    let input32: Vec<Complex32> = input64
        .iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect();
    let mut buf64: [Complex64; 100] = input64.as_slice().try_into().unwrap();
    let mut buf32: [Complex32; 100] = input32.as_slice().try_into().unwrap();
    dft100_impl(&mut buf64, false);
    dft100_impl(&mut buf32, false);
    let err = buf32
        .iter()
        .zip(buf64.iter())
        .map(|(a, b)| {
            let diff = Complex32::new((a.re - b.re as f32).abs(), (a.im - b.im as f32).abs());
            diff.re.max(diff.im)
        })
        .fold(0.0f32, f32::max);
    assert!(err < 5e-4, "f32 vs f64 DFT-100 max_err={err:.2e}");
}

// ── DFT-6 ─────────────────────────────────────────────────────────────────────

macro_rules! composite_tests {
    ($name:ident, $n:literal, $fn:ident, $tol:expr, $ops:expr) => {
        mod $name {
            use super::*;
            #[test]
            fn forward_matches_direct() {
                let input: Vec<Complex64> = (0..$n)
                    .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.27).cos()))
                    .collect();
                let expected = dft_forward(&input);
                let mut buf: [Complex64; $n] = input.as_slice().try_into().unwrap();
                $fn(&mut buf, false);
                let err = max_err(&buf, &expected);
                let bound = roundoff_bound(&input, $ops);
                assert!(err <= bound, "forward max_err={err:.2e}, bound={bound:.2e}");
            }
            #[test]
            fn inverse_matches_direct() {
                let input: Vec<Complex64> = (0..$n)
                    .map(|k| Complex64::new((k as f64 * 0.33).cos(), (k as f64 * 0.61).sin()))
                    .collect();
                let expected: Vec<Complex64> = dft_inverse(&input)
                    .into_iter()
                    .map(|x| x * $n as f64)
                    .collect();
                let mut buf: [Complex64; $n] = input.as_slice().try_into().unwrap();
                $fn(&mut buf, true);
                let err = max_err(&buf, &expected);
                let bound = roundoff_bound(&input, $ops);
                assert!(err <= bound, "inverse max_err={err:.2e}, bound={bound:.2e}");
            }
            #[test]
            fn roundtrip_recovers_input() {
                let input: Vec<Complex64> = (0..$n)
                    .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.53).cos()))
                    .collect();
                let mut buf: [Complex64; $n] = input.as_slice().try_into().unwrap();
                $fn(&mut buf, false);
                $fn(&mut buf, true);
                let recovered: Vec<Complex64> = buf.iter().map(|x| x / $n as f64).collect();
                let err = max_err(&recovered, &input);
                let bound = roundoff_bound(&input, $ops * 2);
                assert!(
                    err <= bound,
                    "roundtrip max_err={err:.2e}, bound={bound:.2e}"
                );
            }
            #[test]
            fn dc_energy_in_bin0_only() {
                let mut buf = [Complex64::new(1.0, 0.0); $n];
                $fn(&mut buf, false);
                let bound = roundoff_bound(&buf, $ops);
                assert!(
                    (buf[0] - Complex64::new($n as f64, 0.0)).norm() <= bound,
                    "DC bin: {:?}",
                    buf[0]
                );
                for (k, x) in buf[1..].iter().enumerate() {
                    assert!(x.norm() <= bound, "non-zero bin[{}]: {:?}", k + 1, x);
                }
            }
            #[test]
            fn f32_forward_matches_f64() {
                let input64: Vec<Complex64> = (0..$n)
                    .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.27).cos()))
                    .collect();
                let input32: Vec<Complex32> = input64
                    .iter()
                    .map(|c| Complex32::new(c.re as f32, c.im as f32))
                    .collect();
                let mut buf64: [Complex64; $n] = input64.as_slice().try_into().unwrap();
                let mut buf32: [Complex32; $n] = input32.as_slice().try_into().unwrap();
                $fn(&mut buf64, false);
                $fn(&mut buf32, false);
                let err = buf32
                    .iter()
                    .zip(buf64.iter())
                    .map(|(a, b)| {
                        let diff =
                            Complex32::new((a.re - b.re as f32).abs(), (a.im - b.im as f32).abs());
                        diff.re.max(diff.im)
                    })
                    .fold(0.0f32, f32::max);
                assert!(err < $tol, "f32 vs f64 max_err={err:.2e}");
            }
        }
    };
}

composite_tests!(dft6_tests, 6, dft6_impl, 1e-4_f32, 100);
composite_tests!(dft9_tests, 9, dft9_impl, 1e-4_f32, 200);
composite_tests!(dft10_tests, 10, dft10_impl, 1e-4_f32, 200);
composite_tests!(dft12_tests, 12, dft12_impl, 1e-4_f32, 300);
composite_tests!(dft14_tests, 14, dft14_impl, 1e-4_f32, 400);
composite_tests!(dft18_tests, 18, dft18_impl, 1e-4_f32, 500);
composite_tests!(dft22_tests, 22, dft22_impl, 1e-4_f32, 600);
composite_tests!(dft28_tests, 28, dft28_impl, 1e-4_f32, 800);
composite_tests!(dft30_tests, 30, dft30_impl, 1e-4_f32, 900);
composite_tests!(dft36_tests, 36, dft36_impl, 1e-4_f32, 1200);
composite_tests!(dft42_tests, 42, dft42_impl, 1e-4_f32, 1400);
composite_tests!(dft45_tests, 45, dft45_impl, 1e-4_f32, 1500);
composite_tests!(dft48_tests, 48, dft48_impl, 1e-4_f32, 1600);
composite_tests!(dft63_tests, 63, dft63_impl, 1e-4_f32, 2000);
composite_tests!(dft33_tests, 33, dft33_impl, 1e-4_f32, 1100);
composite_tests!(dft35_tests, 35, dft35_impl, 1e-4_f32, 1200);
composite_tests!(dft40_tests, 40, dft40_impl, 1e-4_f32, 1300);
composite_tests!(dft50_tests, 50, dft50_impl, 1e-4_f32, 1800);
composite_tests!(dft56_tests, 56, dft56_impl, 1e-4_f32, 2000);
composite_tests!(dft70_tests, 70, dft70_impl, 1e-4_f32, 2500);
composite_tests!(dft72_tests, 72, dft72_impl, 1e-4_f32, 2600);
composite_tests!(dft75_tests, 75, dft75_impl, 1e-4_f32, 2700);
composite_tests!(dft77_tests, 77, dft77_impl, 1e-4_f32, 2800);
composite_tests!(dft80_tests, 80, dft80_impl, 1e-4_f32, 2900);
composite_tests!(dft84_tests, 84, dft84_impl, 1e-4_f32, 3000);
composite_tests!(dft88_tests, 88, dft88_impl, 1e-4_f32, 3200);
composite_tests!(dft90_tests, 90, dft90_impl, 1e-4_f32, 3300);
composite_tests!(dft96_tests, 96, dft96_impl, 1e-4_f32, 3500);
composite_tests!(dft98_tests, 98, dft98_impl, 1e-4_f32, 3600);
composite_tests!(dft99_tests, 99, dft99_impl, 1e-4_f32, 3600);
composite_tests!(dft105_tests, 105, dft105_impl, 1e-4_f32, 3900);
composite_tests!(dft112_tests, 112, dft112_impl, 1e-4_f32, 4100);
composite_tests!(dft120_tests, 120, dft120_impl, 1e-4_f32, 4400);
composite_tests!(dft126_tests, 126, dft126_impl, 1e-4_f32, 4700);
composite_tests!(dft128_tests, 128, dft128_impl, 1e-4_f32, 4800);
composite_tests!(dft49_tests, 49, dft49_impl, 1e-4_f32, 1600);
