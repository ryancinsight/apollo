//! Published-reference fixtures for the FFT transform family.

#![expect(
    clippy::unwrap_used,
    reason = "ratchet APOLLO-UNWRAP-1: pre-existing debt"
)]

use super::builders::{published_complex_fixture, published_real_fixture};
use crate::domain::report::PublishedFixtureReport;
use eunomia::Complex64;
use leto::Array1;
use leto::Storage;

pub(crate) fn fft_four_point_difference_fixture() -> PublishedFixtureReport {
    let signal_nd = Array1::from(vec![1.0, 0.0, -1.0, 0.0]);
    let signal = leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
        [signal_nd.size()],
        signal_nd.as_slice().unwrap(),
    )
    .unwrap();
    let actual_leto = apollo_fft::fft_1d_leto(signal.view());
    let actual = leto::Array1::from(actual_leto.storage().as_slice().to_vec());
    let expected = [
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    published_complex_fixture(
        "FFT",
        "DFT4([1,0,-1,0])",
        "Cooley and Tukey (1965), finite root-of-unity DFT definition",
        actual.iter(),
        expected.iter(),
    )
}

/// IDFT of the all-ones vector [1,1,1,1] with N=4.
///
/// # Mathematical contract
///
/// The normalized IDFT is IDFT[n] = (1/N) Σ_k F[k] exp(2πikn/N).
/// For F=[1,1,1,1] and N=4, every sum collapses to IDFT[0]=1, IDFT[1..3]=0
/// because Σ_k exp(2πikn/N) = N·δ_{n,0} by the geometric series identity for
/// primitive roots of unity. Reference: DFT Inversion Theorem, Cooley and Tukey (1965).
pub(crate) fn fft_inverse_four_point_fixture() -> PublishedFixtureReport {
    let spectrum_nd = Array1::from(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ]);
    let spectrum = leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
        [spectrum_nd.size()],
        spectrum_nd.as_slice().unwrap(),
    )
    .unwrap();
    let actual_leto = apollo_fft::ifft_1d_leto(spectrum.view());
    let actual = leto::Array1::from(actual_leto.storage().as_slice().to_vec());
    let expected = [1.0_f64, 0.0, 0.0, 0.0];
    published_real_fixture(
        "FFT",
        "IDFT4([1,1,1,1])",
        "Cooley and Tukey (1965), DFT inversion theorem: IDFT(DFT(x))=x; DFT([1,0,0,0])=[1,1,1,1] so IDFT([1,1,1,1])=[1,0,0,0]",
        actual.as_slice().unwrap(),
        &expected,
    )
}
