use crate::application::execution::kernel::components::rader;
use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use eunomia::{Complex32, Complex64};

fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max)
}

fn signal(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|k| Complex64::new((k as f64 * 0.19).sin(), (k as f64 * 0.37).cos()))
        .collect()
}

fn assert_short_forward_precise<const N: usize>(kernel: impl FnOnce(&mut [Complex64; N])) {
    let input = signal(N);
    let expected = dft_forward(&input);
    let mut buf: [Complex64; N] = input.as_slice().try_into().unwrap();
    kernel(&mut buf);
    let err = max_err(&buf, &expected);
    let tol = 512.0 * N as f64 * f64::EPSILON;
    assert!(
        err < tol,
        "DFT-{N} f64 forward max_err={err:.2e}, tol={tol:.2e}"
    );
}

fn assert_short_forward_reduced<const N: usize>(kernel: impl FnOnce(&mut [Complex32; N])) {
    let input = signal(N);
    let expected = dft_forward(&input);
    let mut buf: [Complex32; N] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    kernel(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    let tol = 32.0 * N as f64 * f32::EPSILON as f64;
    assert!(
        err < tol,
        "DFT-{N} f32 forward max_err={err:.2e}, tol={tol:.2e}"
    );
}

fn assert_short_inverse_reduced<const N: usize>(kernel: impl FnOnce(&mut [Complex32; N])) {
    let input = signal(N);
    let expected: Vec<_> = dft_inverse(&input)
        .into_iter()
        .map(|x| x * N as f64)
        .collect();
    let mut buf: [Complex32; N] =
        core::array::from_fn(|i| Complex32::new(input[i].re as f32, input[i].im as f32));
    kernel(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    let tol = 32.0 * N as f64 * f32::EPSILON as f64;
    assert!(
        err < tol,
        "DFT-{N} f32 inverse max_err={err:.2e}, tol={tol:.2e}"
    );
}

#[test]
fn generated_rader_primes_match_direct_forward_and_inverse() {
    for n in [
        17usize, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 109,
        113, 127, 131, 151, 181, 193, 197, 199,
    ] {
        let input = signal(n);

        let expected_forward = dft_forward(&input);
        let mut forward = input.clone();
        rader::rader_fft::<f64, false>(&mut forward);
        let forward_err = max_err(&forward, &expected_forward);
        assert!(
            forward_err < 8e-11,
            "N={n} generated Rader forward max_err={forward_err:.2e}"
        );

        let expected_inverse: Vec<_> = dft_inverse(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect();
        let mut inverse = input.clone();
        rader::rader_fft::<f64, true>(&mut inverse);
        let inverse_err = max_err(&inverse, &expected_inverse);
        assert!(
            inverse_err < 8e-11,
            "N={n} generated Rader inverse max_err={inverse_err:.2e}"
        );
    }
}

#[test]
fn dft17_forward_matches_direct() {
    let input: Vec<Complex64> = (0..17)
        .map(|k| Complex64::new((k as f64 * 0.19).sin(), (k as f64 * 0.37).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: [Complex64; 17] = input.as_slice().try_into().unwrap();
    <f64 as ShortWinogradScalar>::dft17::<false>(&mut buf);
    let err = max_err(&buf, &expected);
    assert!(err < 2e-12, "DFT-17 forward max_err={err:.2e}");
}

#[test]
fn dft17_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..17)
        .map(|k| Complex64::new((k as f64 * 0.29).cos(), -(k as f64 * 0.13).sin()))
        .collect();
    let mut buf: [Complex64; 17] = input.as_slice().try_into().unwrap();
    <f64 as ShortWinogradScalar>::dft17::<false>(&mut buf);
    <f64 as ShortWinogradScalar>::dft17::<true>(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| *x / 17.0).collect();
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
    <f64 as ShortWinogradScalar>::dft17::<true>(&mut buf);
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
    <f32 as ShortWinogradScalar>::dft17::<false>(&mut buf);
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
    <f64 as ShortWinogradScalar>::dft23::<false>(&mut buf);
    let err = max_err(&buf, &expected);
    assert!(err < 4e-12, "DFT-23 forward max_err={err:.2e}");
}

#[test]
fn dft23_inverse_roundtrip() {
    let input: Vec<Complex64> = (0..23)
        .map(|k| Complex64::new((k as f64 * 0.27).cos(), -(k as f64 * 0.11).sin()))
        .collect();
    let mut buf: [Complex64; 23] = input.as_slice().try_into().unwrap();
    <f64 as ShortWinogradScalar>::dft23::<false>(&mut buf);
    <f64 as ShortWinogradScalar>::dft23::<true>(&mut buf);
    let recovered: Vec<Complex64> = buf.iter().map(|x| *x / 23.0).collect();
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
    <f64 as ShortWinogradScalar>::dft23::<true>(&mut buf);
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
    <f32 as ShortWinogradScalar>::dft23::<false>(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 4e-5, "DFT-23 f32 forward max_err={err:.2e}");
}

#[test]
fn promoted_short_odd_prime_f64_routes_match_direct() {
    assert_short_forward_precise::<29>(<f64 as ShortWinogradScalar>::dft29::<false>);
    assert_short_forward_precise::<31>(<f64 as ShortWinogradScalar>::dft31::<false>);
    assert_short_forward_precise::<37>(<f64 as ShortWinogradScalar>::dft37::<false>);
    assert_short_forward_precise::<41>(<f64 as ShortWinogradScalar>::dft41::<false>);
    assert_short_forward_precise::<43>(<f64 as ShortWinogradScalar>::dft43::<false>);
    assert_short_forward_precise::<47>(<f64 as ShortWinogradScalar>::dft47::<false>);
    assert_short_forward_precise::<53>(<f64 as ShortWinogradScalar>::dft53::<false>);
}

#[test]
fn short_odd_prime_f32_routes_match_direct() {
    assert_short_forward_reduced::<11>(<f32 as ShortWinogradScalar>::dft11::<false>);
    assert_short_forward_reduced::<13>(<f32 as ShortWinogradScalar>::dft13::<false>);
    assert_short_forward_reduced::<17>(<f32 as ShortWinogradScalar>::dft17::<false>);
    assert_short_forward_reduced::<19>(<f32 as ShortWinogradScalar>::dft19::<false>);
    assert_short_forward_reduced::<23>(<f32 as ShortWinogradScalar>::dft23::<false>);
    assert_short_forward_reduced::<29>(<f32 as ShortWinogradScalar>::dft29::<false>);
    assert_short_forward_reduced::<31>(<f32 as ShortWinogradScalar>::dft31::<false>);
    assert_short_forward_reduced::<37>(<f32 as ShortWinogradScalar>::dft37::<false>);
    assert_short_forward_reduced::<41>(<f32 as ShortWinogradScalar>::dft41::<false>);
    assert_short_forward_reduced::<43>(<f32 as ShortWinogradScalar>::dft43::<false>);
    assert_short_forward_reduced::<47>(<f32 as ShortWinogradScalar>::dft47::<false>);
    assert_short_forward_reduced::<53>(<f32 as ShortWinogradScalar>::dft53::<false>);
}

#[test]
fn reduced_short_odd_prime_f32_inverse_route_matches_direct() {
    assert_short_inverse_reduced::<31>(<f32 as ShortWinogradScalar>::dft31::<true>);
}

/// Rader's algorithm for N=10007 (large prime, recursive chain 10007→5003→{41,61}).
///
/// Direct DFT reference is O(N²) and too expensive, so we verify via roundtrip:
/// forward ∘ inverse (unnormalized) ∘ (1/N) = identity.
///
/// Spawns on a dedicated thread with 8MB stack because the Rader runtime path
/// allocates two stack arrays of size (N-1) × Complex64 ≈ 2 × 160 KB.
#[test]
fn rader_fft_n10007_roundtrip() {
    let handle = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let n = 10007usize;
            let input = signal(n);
            let mut buf = input.clone();
            rader::rader_fft::<f64, false>(&mut buf);
            rader::rader_fft::<f64, true>(&mut buf);
            let recovered: Vec<_> = buf.iter().map(|x| *x / n as f64).collect();
            let err = max_err(&recovered, &input);
            assert!(err < 1e-8, "Rader N=10007 roundtrip max_err={err:.2e}");
        })
        .expect("failed to spawn test thread");
    handle.join().expect("test thread panicked");
}
