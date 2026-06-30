//! Published-reference fixtures for the STFT transform family.

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

pub(crate) fn stft_rectangular_window_impulse_frame_fixture() -> SuiteResult<PublishedFixtureReport>
{
    // Signal: unit impulse at position 0, frame_len=4, hop_len=4, rectangular window.
    // Centered framing: frame 0 center at 0, starts at index -2.
    // Frame 0 reads signal indices [-2,-1,0,1]: out-of-bounds pad to 0.
    //   Windowed frame 0 = [0, 0, x[0], x[1]] = [0, 0, 1, 0].
    // DFT([0,0,1,0]): X[k] = exp(-2πi·2·k/4) = exp(-πik) = (-1)^k.
    //   X[0]=1, X[1]=-1, X[2]=1, X[3]=-1.
    // Frame 1 reads signal indices [2,3,4,5]: indices 4,5 out of bounds.
    //   Windowed frame 1 = [x[2],x[3],0,0] = [0,0,0,0]; DFT = [0,0,0,0].
    // Full output: [1,-1,1,-1, 0,0,0,0].
    // Reference: Cooley-Tukey (1965) DFT shift theorem X[k]=exp(-2πikn₀/N) for δ[n-n₀];
    //            Allen & Rabiner (1977) STFT centered-frame analysis.
    let plan = StftPlan::new(4, 4)?;
    let signal = Array1::from(vec![1.0_f64, 0.0, 0.0, 0.0]);
    let window = [1.0_f64, 1.0, 1.0, 1.0];
    let output = plan.forward_with_window(&signal, &window)?;
    let actual: Vec<Complex64> = output.iter().copied().collect();
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "STFT",
        "STFT-rect-window-impulse-centered-frame(N=4)",
        "Cooley-Tukey (1965) DFT shift: X[k]=(-1)^k for δ[n-2] in 4-point frame; Allen & Rabiner (1977) STFT centered framing",
        actual.iter(),
        expected.iter(),
    ))
}

/// STFT Hann-window WOLA inverse roundtrip: ISTFT(STFT(x)) = x.
///
/// # Mathematical contract
///
/// Parameters: frame_len=4, hop_len=2, Hann analysis+synthesis window.
/// Hann window N=4: w[n] = 0.5·(1 − cos(2πn/3)), giving w=[0, 0.75, 0.75, 0].
/// Signal x=[1,0,0,0]. frame_count(4) = 1 + ⌈4/2⌉ = 3 frames.
///
/// Frame centers (start = m·hop − frame/2):
///   m=0: start=−2, covers signal positions 0,1 (frame indices 2,3).
///     Windowed frame: [0,0, w[2]·x[0], w[3]·x[1]] = [0,0, 0.75, 0].
///     DFT([0,0,0.75,0]) = [0.75, −0.75, 0.75, −0.75].
///   m=1: start=0, covers positions 0..3. Windowed: [w[0]·1,0,0,0]=[0,0,0,0]. Spectrum=[0,…].
///   m=2: start=2, covers positions 2..5. All zero. Spectrum=[0,…].
///
/// Reconstruction (WOLA): for position i, output[i] = Σ_m (w[n_m]·IFFT(X_m)[n_m]) / Σ_m w[n_m]².
///   pos 0: IFFT([0.75,−0.75,0.75,−0.75])[2] = 0.75; weight = w[2]² = 0.5625.
///     output[0] = 0.5625 / 0.5625 = 1.0 ✓
///   pos 1: weight = w[3]² = 0. output[1] = 0.0 ✓  (WOLA defines 0/0 → 0)
///   pos 2,3: no non-zero contributions. output = 0.0 ✓
///
/// COLA weight at every position: 0.5625 (constant) for non-boundary positions.
/// Boundary position 1 has w[3]=0; expected x[1]=0; WOLA returns 0 by convention. ✓
/// Floating-point error: O(log₂(4)·ε_f64) ≈ 4.4×10⁻¹⁶ < 1×10⁻¹².
/// Reference: Allen & Rabiner (1977) Proc. IEEE 65(11): weighted overlap-add synthesis;
///            Hann window COLA condition, Portnoff (1980) IEEE Trans. ASSP 28(1).
pub(crate) fn stft_hann_wola_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = StftPlan::new(4, 2)?;
    let signal = Array1::from(vec![1.0_f64, 0.0, 0.0, 0.0]);
    let spectrum = plan.forward(&signal)?;
    let recovered = plan.inverse(&spectrum, 4)?;
    let expected = [1.0_f64, 0.0, 0.0, 0.0];
    Ok(published_real_fixture_with_threshold(
        "STFT",
        "STFT-Hann-WOLA-inverse-roundtrip([1,0,0,0],frame=4,hop=2)",
        "Allen & Rabiner (1977) Proc. IEEE 65(11): WOLA synthesis; Portnoff (1980) Hann COLA; ISTFT(STFT(x))=x for frame=4 hop=2",
        recovered.as_slice().unwrap_or(&[]),
        &expected,
        1.0e-12,
    ))
}
