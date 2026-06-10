//! Published-reference fixtures for the SFT transform family.

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
use ndarray::{Array1, Array2};
use num_complex::Complex64;

pub(crate) fn sft_one_sparse_alternating_tone_fixture() -> SuiteResult<PublishedFixtureReport> {
    // x[n] = (-1)^n for N=4 is a single-frequency pure tone.
    // DFT: X[k] = N·δ(k - N/2) = 4·δ(k-2) (Parseval shift theorem).
    // SparseFftPlan::new(4,1) retains the 1 largest-magnitude coefficient.
    // Exact recovery theorem: top-1 heap finds bin 2 with value 4+0i;
    // all other bins have magnitude 0 and cannot displace the winner.
    // to_dense() expands to [0+0i, 0+0i, 4+0i, 0+0i].
    // Reference: Cooley-Tukey (1965) DFT[(−1)^n]=N·δ[k−N/2];
    //            Hassanieh et al. (2012) sFFT exact K-sparse recovery theorem.
    let plan = SparseFftPlan::new(4, 1)?;
    let signal = [
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    let spectrum = plan.forward(&signal)?;
    let dense = spectrum.to_dense();
    let expected = [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "SFT",
        "SFT-1-sparse-alternating-tone(N=4,K=1)",
        "Cooley-Tukey (1965): DFT[(−1)^n]=4·δ[k−2] for N=4; Hassanieh et al. (2012) sFFT exact K-sparse recovery",
        dense.iter(),
        expected.iter(),
    ))
}

/// SFT 1-sparse alternating tone inverse roundtrip: ISFT(SFT(x)) = x.
///
/// # Mathematical contract
///
/// Signal x[n] = (-1)^n for N=4 is a pure tone at frequency k=2.
/// DFT: X[k] = N·δ(k-N/2) = 4·δ(k-2), so X = [0,0,4,0].
/// SparseFftPlan::new(4,1) retains K=1 largest coefficient: freq=2, value=4+0i.
/// SparseSpectrum is exactly [0,0,4+0i,0] when expanded.
/// ISFT expands to dense [0,0,4,0] then applies IFFT normalized by 1/N:
///   x_rec[n] = (1/4)·Σ_k X[k]·e^{2πink/N} = (1/4)·4·e^{iπn} = e^{iπn} = [1,-1,1,-1]. ✓
/// Threshold 1×10⁻¹² covers IEEE 754 rounding of e^{iπn} for n=0..3.
/// Reference: Cooley-Tukey (1965): DFT[(−1)^n]=N·δ[k−N/2];
///            Hassanieh et al. (2012) sFFT exact K-sparse recovery;
///            Candès & Wakin (2008) RIP exact reconstruction theorem.
pub(crate) fn sft_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = SparseFftPlan::new(4, 1)?;
    let signal = [
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    let spectrum = plan.forward(&signal)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_complex_fixture(
        "SFT",
        "SFT-inverse-roundtrip-1-sparse-alternating-tone(N=4,K=1)",
        "Cooley-Tukey (1965) DFT[(−1)^n]=4·δ[k−2]; Hassanieh et al. (2012) K-sparse exact recovery; ISFT(SFT(x))=x",
        recovered.iter(),
        signal.iter(),
    ))
}
