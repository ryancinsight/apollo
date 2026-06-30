//! Published-reference fixtures for the HILBERT transform family.

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

pub(crate) fn hilbert_cosine_to_sine_fixture() -> SuiteResult<PublishedFixtureReport> {
    // H{cos(2πn/N)} = sin(2πn/N) for bin-frequency f₀=1, N=4.
    // Input: x[n] = cos(2πn/4) = [1, 0, -1, 0].
    // DFT mask: multiply positive bins by -i, DC/Nyquist by 0, negative bins by +i.
    // DFT(x) = [0, 2, 0, 2]; after mask: [0, -2i, 0, 2i].
    // IDFT([0,-2i,0,2i]) = [0, 1, 0, -1] = sin(2πn/4). ✓
    // Reference: Bracewell (1965) §12: H{cos(ω₀n)} = sin(ω₀n) for single-bin frequency;
    //            Oppenheim & Schafer (1999) §12.1 discrete Hilbert via DFT mask.
    let plan = HilbertPlan::new(4)?;
    let signal = [1.0_f64, 0.0, -1.0, 0.0];
    let quadrature = plan.transform(&signal)?;
    let expected = [0.0_f64, 1.0, 0.0, -1.0];
    Ok(published_real_fixture(
        "Hilbert",
        "Hilbert-cosine-to-sine-4point",
        "Bracewell (1965) §12: H{cos(2πn/N)}=sin(2πn/N); Oppenheim-Schafer (1999) §12.1 discrete Hilbert via DFT mask",
        &quadrature,
        &expected,
    ))
}

pub(crate) fn hilbert_instantaneous_frequency_constant_tone_fixture(
) -> SuiteResult<PublishedFixtureReport> {
    // Instantaneous frequency of a discrete cosine at normalised frequency f₀ = k/N.
    // The analytic signal of cos(2πkn/N) is exp(2πi·k·n/N), so:
    //   f[n] = arg(conj(z[n]) · z[n+1]) / (2π) = k/N (constant).
    // Reference: Boashash (1992) "Estimating and interpreting the instantaneous
    //            frequency of a signal", Proc. IEEE 80(4): §II.A complex-derivative
    //            formula f(t) = (1/2π) d/dt arg(z(t)).
    const TOL: f64 = 1.0e-10;
    let n: usize = 64;
    let k: usize = 5;
    let f_expected = k as f64 / n as f64;
    let signal: Vec<f64> = (0..n)
        .map(|i| (std::f64::consts::TAU * k as f64 * i as f64 / n as f64).cos())
        .collect();
    let plan = HilbertPlan::new(n)?;
    let analytic = plan.analytic_signal(&signal)?;
    let freq = analytic.instantaneous_frequency();
    // freq has length N-1; compare each sample to f_expected
    let expected: Vec<f64> = vec![f_expected; freq.len()];
    Ok(published_real_fixture_with_threshold(
        "Hilbert",
        "Hilbert-instantaneous-frequency-tone(N=64,k=5)",
        "Boashash (1992) §II.A: IF of cos(2πkn/N) is k/N via complex-derivative formula",
        &freq,
        &expected,
        TOL,
    ))
}

/// Hilbert envelope of a pure discrete cosine tone is identically unity.
///
/// # Mathematical contract
///
/// For a real bandpass signal x[n] = A·cos(ω₀n + φ), the analytic signal
/// z[n] = x[n] + i·H{x}[n] = A·exp(i(ω₀n + φ)) and the instantaneous
/// envelope |z[n]| = A (constant). (Oppenheim & Schafer 2010 §12.1, eq.(12.8))
///
/// Analytical derivation for x=[1,0,-1,0] = cos(πn/2), N=4:
///   DFT(x): X[0]=0, X[1]=2, X[2]=0, X[3]=2
///   Hilbert analytic mask for N=4 (even): [1, 2, 1, 0]
///   Y = X ⊙ mask = [0, 4, 0, 0]
///   z = IDFT(Y)/1 = exp(iπn/2): z[0]=1, z[1]=i, z[2]=-1, z[3]=-i
///   |z[0]|=1, |z[1]|=1, |z[2]|=1, |z[3]|=1
///
/// All DFT factors ∈{1,i,-1,-i} and mask values ∈{0,1,2}; the envelope
/// vector [1,1,1,1] is an exact integer result. Accumulated FP error is
/// O(log₂(N)·ε_f64) ≈ 8.9×10^{-16} << threshold 1×10^{-12}.
///
/// Reference: Oppenheim, A.V. & Schafer, R.W. (2010).
///            Discrete-Time Signal Processing (3rd ed.) §12.1, eq.(12.8).
///            Bedrosian, E. (1963) Proc. IEEE 51(5): analytic signal envelope theorem.
pub(crate) fn hilbert_pure_cosine_envelope_is_unity_fixture() -> SuiteResult<PublishedFixtureReport>
{
    let signal = [1.0_f64, 0.0, -1.0, 0.0];
    let plan = HilbertPlan::new(4)?;
    let envelope = plan.envelope(&signal)?;
    let expected = [1.0_f64, 1.0, 1.0, 1.0];
    Ok(published_real_fixture_with_threshold(
        "Hilbert",
        "Hilbert-envelope(cos(πn/2),N=4)=[1,1,1,1]",
        "Oppenheim & Schafer (2010) DTSP 3rd ed. §12.1 eq.(12.8): |z[n]|=A for x[n]=A·cos(ω₀n+φ); Bedrosian (1963)",
        &envelope,
        &expected,
        1.0e-12,
    ))
}
