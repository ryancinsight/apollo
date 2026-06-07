//! Published-reference fixtures for the DHT transform family.

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

pub(crate) fn dht_four_point_difference_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DhtPlan::new(4)?;
    let spectrum = plan.forward(&[1.0, 0.0, -1.0, 0.0])?;
    let expected = [0.0, 2.0, 0.0, 2.0];
    Ok(published_real_fixture(
        "DHT",
        "DHT4([1,0,-1,0])",
        "Bracewell (1983), cas(theta)=cos(theta)+sin(theta) Hartley definition",
        spectrum.values(),
        &expected,
    ))
}

/// DHT self-reciprocal property: DHT(DHT(x)) = N·x.
///
/// # Mathematical contract
///
/// For x=[1,0,0,0] and N=4:
///   step1 = DHT([1,0,0,0]) = [1,1,1,1]  (cas impulse response at n=0)
///   step2 = DHT([1,1,1,1]) = [4,0,0,0]  (DC-only signal maps to scaled impulse)
/// step2 = N·x = 4·[1,0,0,0] = [4,0,0,0]. ✓
/// Reference: Bracewell (1983), DHT self-reciprocal property: H{H{x}}[n] = N·x[n].
pub(crate) fn dht_self_reciprocal_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DhtPlan::new(4)?;
    let step1 = plan.forward(&[1.0_f64, 0.0, 0.0, 0.0])?;
    let step2 = plan.forward(step1.values())?;
    let expected = [4.0_f64, 0.0, 0.0, 0.0];
    Ok(published_real_fixture(
        "DHT",
        "DHT(DHT([1,0,0,0]))=[4,0,0,0]",
        "Bracewell (1983), DHT self-reciprocal property: DHT(DHT(x))=N\u{00b7}x; for x=[1,0,0,0], DHT([1,0,0,0])=[1,1,1,1] then DHT([1,1,1,1])=[4,0,0,0]",
        step2.values(),
        &expected,
    ))
}

/// DHT 4-point inverse roundtrip: IDHT(DHT(x)) = x.
///
/// # Mathematical contract
///
/// The DHT kernel: H[k] = Σ_{n=0}^{N-1} x[n] · cas(2πnk/N), cas(t) = cos(t) + sin(t).
/// For the inverse: IDHT(X)[n] = (1/N) · DHT(X)[n] (self-reciprocal property, H²=NI).
/// For x=[3,-1,2,0] and N=4:
///   H[0] = 3-1+2+0 = 4
///   H[1] = 3·cas(0) + (-1)·cas(π/2) + 2·cas(π) + 0·cas(3π/2) = 3-1-2 = 0
///   H[2] = 3·cas(0) + (-1)·cas(π) + 2·cas(2π) + 0·cas(3π) = 3+1+2 = 6
///   H[3] = 3·cas(0) + (-1)·cas(3π/2) + 2·cas(3π) + 0·cas(9π/2) = 3+1-2 = 2
/// DHT([3,-1,2,0]) = [4,0,6,2].
/// IDHT([4,0,6,2]) = DHT([4,0,6,2]) / 4 = [12,-4,8,0] / 4 = [3,-1,2,0]. ✓
/// Roundtrip error bounded by butterfly accumulation: O(log₂N · ε_f64) < 1×10⁻¹⁴ for N=4.
/// Reference: Bracewell (1983) JOSA 73(12): DHT self-reciprocal property H²=NI; inverse = (1/N)·DHT.
pub(crate) fn dht_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DhtPlan::new(4)?;
    let input = [3.0_f64, -1.0, 2.0, 0.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DHT",
        "DHT-inverse-roundtrip([3,-1,2,0],N=4)",
        "Bracewell (1983) JOSA 73(12): DHT self-reciprocal H\u{00b2}=NI; IDHT(DHT(x))=x; x=[3,-1,2,0]",
        &recovered,
        &input,
        1.0e-14,
    ))
}
