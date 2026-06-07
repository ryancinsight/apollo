//! Published-reference fixtures for the FFT transform family.

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
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

pub(crate) fn fft_four_point_difference_fixture() -> PublishedFixtureReport {
    let signal = Array1::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
    let actual = fft_1d_array(&signal);
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
    let spectrum = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ]);
    let actual = ifft_1d_array(&spectrum);
    let expected = [1.0_f64, 0.0, 0.0, 0.0];
    published_real_fixture(
        "FFT",
        "IDFT4([1,1,1,1])",
        "Cooley and Tukey (1965), DFT inversion theorem: IDFT(DFT(x))=x; DFT([1,0,0,0])=[1,1,1,1] so IDFT([1,1,1,1])=[1,0,0,0]",
        actual.as_slice().unwrap(),
        &expected,
    )
}
