//! Published-reference fixtures for the QFT transform family.

#![allow(unused_imports)]

use super::super::SuiteResult;
use super::builders::{
    published_complex_fixture, published_real_fixture, published_real_fixture_with_threshold,
};
use crate::domain::report::PublishedFixtureReport;
use apollo_czt::CztPlan;
use apollo_dctdst::{DctDstPlan, RealTransformKind};
use apollo_dht::DhtPlan;
use apollo_fft::{fft_1d_array, ifft_1d_array};
use apollo_frft::UnitaryFrftPlan;
use apollo_fwht::FwhtPlan;
use apollo_gft::GftPlan;
use apollo_hilbert::HilbertPlan;
use apollo_mellin::MellinPlan;
use apollo_ntt::{intt, ntt, NttPlan, DEFAULT_MODULUS};
use apollo_nufft::{nufft_type1_1d, nufft_type2_1d, UniformDomain1D};
use apollo_qft::qft as qft_transform;
use apollo_radon::RadonPlan;
use apollo_sdft::SdftPlan;
use apollo_sft::SparseFftPlan;
use apollo_sht::ShtPlan;
use apollo_stft::StftPlan;
use apollo_wavelet::{ContinuousWavelet, CwtPlan, DiscreteWavelet, DwtPlan};
use leto::{Array1, Array2};
use eunomia::Complex64;

/// QFT two-point known-value fixture: QFT₂(|0⟩) = (1/√2)(|0⟩ + |1⟩).
///
/// # Mathematical contract
///
/// The unitary QFT matrix for N=2 is (1/√2)·[[1,1],[1,−1]].
/// For input [1,0] (computational basis state |0⟩):
///   QFT₂([1,0])ᵀ = (1/√2)·[1,1]ᵀ.
/// Both output components equal 1/√2 = 1/√2 + 0i.
/// Reference: Shor (1994), quantum Fourier transform for N=2 as Hadamard gate.
pub(crate) fn qft_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let actual = qft_transform(&input)?;
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let expected = [
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(inv_sqrt2, 0.0),
    ];
    Ok(published_complex_fixture(
        "QFT",
        "QFT2([1,0])",
        "Shor (1994), QFT for N=2 is the Hadamard gate: QFT([1,0])=[1/\u{221a}2, 1/\u{221a}2]",
        actual.iter(),
        expected.iter(),
    ))
}

/// QFT inverse roundtrip: `iqft(qft(x)) = x` for N=4.
///
/// # Mathematical contract
///
/// Shor (1994) Proc. 35th FOCS §2: QFT_N is unitary, QFT_N† · QFT_N = I_N.
/// The `iqft` free function is the conjugate transpose QFT_N†; both `qft`
/// and `iqft` apply the 1/√N factor (symmetric unitary normalisation). For
/// N=4 (2-qubit computational basis |0000⟩ = [1,0,0,0]), the composition
/// `iqft(qft([1,0,0,0])) = [1,0,0,0]` in exact f64; threshold 1×10⁻¹².
/// Reference: Shor (1994) §2; Nielsen & Chuang (2000) §5.1.
pub(crate) fn qft_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    use apollo_qft::iqft as iqft_fn;
    let input = Array1::from(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let spectrum = qft_transform(&input)?;
    let recovered = iqft_fn(&spectrum)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "QFT",
        "QFT-inverse-roundtrip(N=4,[1,0,0,0])",
        "Shor (1994) sec.2: QFT_N unitary, QFT_N^{dagger}*QFT_N=I; Nielsen & Chuang (2000) sec.5.1",
        recovered.iter(),
        expected.iter(),
    ))
}
