use crate::application::execution::kernel::components::rader;
use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;
use num_complex::{Complex32, Complex64};

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
            let recovered: Vec<_> = buf.iter().map(|x| x / n as f64).collect();
            let err = max_err(&recovered, &input);
            assert!(err < 1e-8, "Rader N=10007 roundtrip max_err={err:.2e}");
        })
        .expect("failed to spawn test thread");
    handle.join().expect("test thread panicked");
}
