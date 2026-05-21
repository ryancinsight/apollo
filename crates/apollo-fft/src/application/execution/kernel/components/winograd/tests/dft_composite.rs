use crate::application::execution::kernel::components::winograd::*;
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
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
    dft15_impl::<_, false>(&mut buf);
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
    dft15_impl::<_, true>(&mut buf);
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-12, "DFT-15 inverse max_err={err:.2e}");
}

#[test]
fn dft15_roundtrip_recovers_input() {
    let input: Vec<Complex64> = (0..15)
        .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.53).cos()))
        .collect();
    let mut buf: [Complex64; 15] = input.as_slice().try_into().unwrap();
    dft15_impl::<_, false>(&mut buf);
    dft15_impl::<_, true>(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 15.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-12, "DFT-15 roundtrip max_err={err:.2e}");
}

#[test]
fn dft15_dc_energy_in_bin0_only() {
    let mut buf = [Complex64::new(1.0, 0.0); 15];
    dft15_impl::<_, false>(&mut buf);
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
    dft15_impl::<_, false>(&mut buf64);
    dft15_impl::<_, false>(&mut buf32);
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
    unsafe {
        dft25_impl::<f64, false>(&mut buf);
    }
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
    unsafe {
        dft25_impl::<f64, true>(&mut buf);
    }
    let err = max_err(&buf, &expected_unnorm);
    assert!(err < 1e-9, "DFT-25 inverse max_err={err:.2e}");
}

#[test]
fn dft25_roundtrip_recovers_input() {
    let input: Vec<Complex64> = (0..25)
        .map(|k| Complex64::new((k as f64 * 0.23).sin(), (k as f64 * 0.47).cos()))
        .collect();
    let mut buf: [Complex64; 25] = input.as_slice().try_into().unwrap();
    unsafe {
        dft25_impl::<f64, false>(&mut buf);
    }
    unsafe {
        dft25_impl::<f64, true>(&mut buf);
    }
    let recovered: Vec<Complex64> = buf.iter().map(|x| x / 25.0).collect();
    let err = max_err(&recovered, &input);
    assert!(err < 1e-9, "DFT-25 roundtrip max_err={err:.2e}");
}

#[test]
fn dft25_dc_energy_in_bin0_only() {
    let mut buf = [Complex64::new(1.0, 0.0); 25];
    unsafe {
        dft25_impl::<f64, false>(&mut buf);
    }
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
    unsafe {
        dft25_impl::<f64, false>(&mut buf64);
    }
    unsafe {
        dft25_impl::<f32, false>(&mut buf32);
    }
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

// ── DFT-6 ─────────────────────────────────────────────────────────────────────

fn run_composite_case<const N: usize>(
    kernel64: impl Fn(&mut [Complex64; N], bool),
    kernel32: impl Fn(&mut [Complex32; N], bool),
    reduced_tol: f32,
    ops: usize,
) {
    let input: Vec<Complex64> = (0..N)
        .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.27).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut forward: [Complex64; N] = input.as_slice().try_into().unwrap();
    kernel64(&mut forward, false);
    let err = max_err(&forward, &expected);
    let bound = roundoff_bound(&input, ops);
    assert!(
        err <= bound,
        "N={N} forward max_err={err:.2e}, bound={bound:.2e}"
    );

    let input: Vec<Complex64> = (0..N)
        .map(|k| Complex64::new((k as f64 * 0.33).cos(), (k as f64 * 0.61).sin()))
        .collect();
    let expected: Vec<Complex64> = dft_inverse(&input)
        .into_iter()
        .map(|x| x * N as f64)
        .collect();
    let mut inverse: [Complex64; N] = input.as_slice().try_into().unwrap();
    kernel64(&mut inverse, true);
    let err = max_err(&inverse, &expected);
    let bound = roundoff_bound(&input, ops);
    assert!(
        err <= bound,
        "N={N} inverse max_err={err:.2e}, bound={bound:.2e}"
    );

    let input: Vec<Complex64> = (0..N)
        .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.53).cos()))
        .collect();
    let mut roundtrip: [Complex64; N] = input.as_slice().try_into().unwrap();
    kernel64(&mut roundtrip, false);
    kernel64(&mut roundtrip, true);
    let recovered: Vec<Complex64> = roundtrip.iter().map(|x| x / N as f64).collect();
    let err = max_err(&recovered, &input);
    let bound = roundoff_bound(&input, ops * 2);
    assert!(
        err <= bound,
        "N={N} roundtrip max_err={err:.2e}, bound={bound:.2e}"
    );

    let mut dc = [Complex64::new(1.0, 0.0); N];
    kernel64(&mut dc, false);
    let bound = roundoff_bound(&dc, ops);
    assert!(
        (dc[0] - Complex64::new(N as f64, 0.0)).norm() <= bound,
        "N={N} DC bin: {:?}",
        dc[0]
    );
    for (k, x) in dc[1..].iter().enumerate() {
        assert!(x.norm() <= bound, "N={N} non-zero bin[{}]: {:?}", k + 1, x);
    }

    let input64: Vec<Complex64> = (0..N)
        .map(|k| Complex64::new((k as f64 * 0.41).sin(), (k as f64 * 0.27).cos()))
        .collect();
    let input32: Vec<Complex32> = input64
        .iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect();
    let mut precise: [Complex64; N] = input64.as_slice().try_into().unwrap();
    let mut reduced: [Complex32; N] = input32.as_slice().try_into().unwrap();
    kernel64(&mut precise, false);
    kernel32(&mut reduced, false);
    let err = reduced
        .iter()
        .zip(precise.iter())
        .map(|(a, b)| {
            let diff = Complex32::new((a.re - b.re as f32).abs(), (a.im - b.im as f32).abs());
            diff.re.max(diff.im)
        })
        .fold(0.0f32, f32::max);
    assert!(err < reduced_tol, "N={N} f32 vs f64 max_err={err:.2e}");
}

#[test]
fn dft_composite_small_cases() {
    run_composite_case::<6>(
        |data, inv| {
            if inv {
                unsafe { dft6_impl::<f64, true>(data) }
            } else {
                unsafe { dft6_impl::<f64, false>(data) }
            }
        },
        |data, inv| {
            if inv {
                unsafe { dft6_impl::<f32, true>(data) }
            } else {
                unsafe { dft6_impl::<f32, false>(data) }
            }
        },
        1e-4_f32,
        80,
    );
    run_composite_case::<9>(
        |data, inv| {
            if inv {
                dft9_impl::<f64, true>(data)
            } else {
                dft9_impl::<f64, false>(data)
            }
        },
        |data, inv| {
            if inv {
                dft9_impl::<f32, true>(data)
            } else {
                dft9_impl::<f32, false>(data)
            }
        },
        1e-4_f32,
        200,
    );
    run_composite_case::<10>(
        |data, inv| {
            if inv {
                unsafe { dft10_impl::<f64, true>(data) }
            } else {
                unsafe { dft10_impl::<f64, false>(data) }
            }
        },
        |data, inv| {
            if inv {
                unsafe { dft10_impl::<f32, true>(data) }
            } else {
                unsafe { dft10_impl::<f32, false>(data) }
            }
        },
        1e-4_f32,
        120,
    );
    run_composite_case::<12>(
        |data, inv| {
            if inv {
                unsafe { dft12_impl::<f64, true>(data) }
            } else {
                unsafe { dft12_impl::<f64, false>(data) }
            }
        },
        |data, inv| {
            if inv {
                unsafe { dft12_impl::<f32, true>(data) }
            } else {
                unsafe { dft12_impl::<f32, false>(data) }
            }
        },
        1e-4_f32,
        160,
    );
    run_composite_case::<14>(
        |data, inv| {
            if inv {
                unsafe { dft14_impl::<f64, true>(data) }
            } else {
                unsafe { dft14_impl::<f64, false>(data) }
            }
        },
        |data, inv| {
            if inv {
                unsafe { dft14_impl::<f32, true>(data) }
            } else {
                unsafe { dft14_impl::<f32, false>(data) }
            }
        },
        1e-4_f32,
        200,
    );
    run_composite_case::<16>(
        |data, inv| {
            if inv {
                dft16_impl::<f64, true>(data)
            } else {
                dft16_impl::<f64, false>(data)
            }
        },
        |data, inv| {
            if inv {
                dft16_impl::<f32, true>(data)
            } else {
                dft16_impl::<f32, false>(data)
            }
        },
        1e-4_f32,
        400,
    );
    run_composite_case::<27>(
        |data, inv| {
            if inv {
                unsafe { dft27_impl::<f64, true>(data) }
            } else {
                unsafe { dft27_impl::<f64, false>(data) }
            }
        },
        |data, inv| {
            if inv {
                unsafe { dft27_impl::<f32, true>(data) }
            } else {
                unsafe { dft27_impl::<f32, false>(data) }
            }
        },
        1e-4_f32,
        800,
    );
    run_composite_case::<32>(
        |data, inv| {
            if inv {
                dft32_impl::<f64, true>(data)
            } else {
                dft32_impl::<f64, false>(data)
            }
        },
        |data, inv| {
            if inv {
                dft32_impl::<f32, true>(data)
            } else {
                dft32_impl::<f32, false>(data)
            }
        },
        1e-4_f32,
        1000,
    );
}

#[test]
fn dft_composite_medium_cases() {
    run_composite_case::<64>(
        |data, inv| {
            if inv {
                dft64_impl::<f64, true>(data)
            } else {
                dft64_impl::<f64, false>(data)
            }
        },
        |data, inv| {
            if inv {
                dft64_impl::<f32, true>(data)
            } else {
                dft64_impl::<f32, false>(data)
            }
        },
        1e-4_f32,
        2000,
    );
    run_composite_case::<49>(
        |data, inv| {
            if inv {
                unsafe { dft49_impl::<f64, true>(data) }
            } else {
                unsafe { dft49_impl::<f64, false>(data) }
            }
        },
        |data, inv| {
            if inv {
                unsafe { dft49_impl::<f32, true>(data) }
            } else {
                unsafe { dft49_impl::<f32, false>(data) }
            }
        },
        1e-4_f32,
        1500,
    );
    run_composite_case::<81>(
        |data, inv| {
            if inv {
                unsafe { dft81_impl::<f64, true>(data) }
            } else {
                unsafe { dft81_impl::<f64, false>(data) }
            }
        },
        |data, inv| {
            if inv {
                unsafe { dft81_impl::<f32, true>(data) }
            } else {
                unsafe { dft81_impl::<f32, false>(data) }
            }
        },
        1e-4_f32,
        3000,
    );
}

#[test]
fn dft_composite_large_cases() {
    run_composite_case::<128>(
        |data, inv| {
            if inv {
                dft128_impl::<f64, true>(data)
            } else {
                dft128_impl::<f64, false>(data)
            }
        },
        |data, inv| {
            if inv {
                dft128_impl::<f32, true>(data)
            } else {
                dft128_impl::<f32, false>(data)
            }
        },
        1e-4_f32,
        4000,
    );
}
