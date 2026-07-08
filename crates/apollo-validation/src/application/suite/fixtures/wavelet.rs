//! Published-reference fixtures for the WAVELET transform family.

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
use eunomia::Complex64;
use leto::{Array1, Array2};

/// Haar DWT one-level detail coefficients for [1, −1, 0, 0].
///
/// # Mathematical contract
///
/// Haar highpass QMF: g = [1/√2, −1/√2] (from g[k] = (−1)^k·h[L−1−k] with h=[1/√2,1/√2]).
/// detail[k] = g[0]·x[2k] + g[1]·x[2k+1] = (x[2k] − x[2k+1]) / √2.
/// Input [1, −1, 0, 0], N=4, levels=1:
///   detail[0] = (1 − (−1)) / √2 = 2/√2 = √2
///   detail[1] = (0 − 0) / √2 = 0
/// Reference: Haar (1910), Mallat (1989) two-channel QMF framework.
pub(crate) fn wavelet_haar_one_level_detail_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, -1.0, 0.0, 0.0];
    let plan = DwtPlan::new(4, 1, DiscreteWavelet::Haar)?;
    let coeffs = plan.forward(&input)?;
    let detail = coeffs.details()[0].as_slice();
    let expected = [std::f64::consts::SQRT_2, 0.0_f64];
    Ok(published_real_fixture(
        "DWT-Haar",
        "Haar-DWT-1level([1,-1,0,0])",
        "Haar (1910), Mallat (1989) two-channel QMF: DWT([1,-1,0,0]) detail=[\u{221a}2,0] via g=[1/\u{221a}2,-1/\u{221a}2]",
        detail,
        &expected,
    ))
}

/// Haar DWT perfect reconstruction: inverse(forward(x)) = x.
///
/// # Mathematical contract
///
/// Mallat (1989) §3.1 Theorem 2 (perfect reconstruction for two-channel QMF banks):
/// if the synthesis filters are the exact duals of the analysis filters, then
/// IDWT(DWT(x)) = x for every finite-energy signal x.  For the Haar pair
/// h = [1/√2, 1/√2] (lowpass) and g = [1/√2, −1/√2] (highpass), this holds
/// exactly in infinite precision.  f64 accumulation error for N=4, 1-level is
/// bounded by O(N·ε_f64) ≈ 4×2.2×10⁻¹⁶ ≈ 9×10⁻¹⁶; threshold is 1×10⁻¹².
/// Reference: Haar (1910); Mallat (1989) §3.1 Theorem 2.
pub(crate) fn wavelet_haar_inverse_perfect_reconstruction_fixture(
) -> SuiteResult<PublishedFixtureReport> {
    let signal = [1.0_f64, -1.0, 0.0, 0.0];
    let plan = DwtPlan::new(4, 1, DiscreteWavelet::Haar)?;
    let coefficients = plan.forward(&signal)?;
    let recovered = plan.inverse(&coefficients)?;
    Ok(published_real_fixture_with_threshold(
        "DWT-Haar",
        "Haar-DWT-inverse-roundtrip(N=4,[1,-1,0,0])",
        "Mallat (1989) §3.1 Theorem 2: perfect reconstruction IDWT(DWT(x))=x for Haar QMF; Haar (1910)",
        &recovered,
        &signal,
        1.0e-12,
    ))
}

/// Daubechies-4 DWT one-level coefficients for x=[1,0,0,0] under periodic extension.
///
/// # Mathematical contract
///
/// For db4 analysis filters (Daubechies 1992, p.198):
///   h = [h0,h1,h2,h3]
///     = [0.4829629131445341,
///        0.8365163037378079,
///        0.2241438680420134,
///       -0.12940952255126034]
/// and QMF highpass g[k] = (−1)^k h[3-k], hence:
///   g = [h3, −h2, h1, −h0]
///     = [−0.12940952255126034,
///        −0.2241438680420134,
///         0.8365163037378079,
///        −0.4829629131445341].
///
/// For N=4, level=1, periodic analysis_stage:
///   a0 = Σ_{m=0}^3 h[m] x[(0+m) mod 4] = h0
///   d0 = Σ_{m=0}^3 g[m] x[(0+m) mod 4] = g0 = h3
///   a1 = Σ_{m=0}^3 h[m] x[(2+m) mod 4] = h2
///   d1 = Σ_{m=0}^3 g[m] x[(2+m) mod 4] = g2 = h1
/// so [a0,a1,d0,d1] = [h0,h2,h3,h1].
///
/// Because x is a basis impulse, each coefficient is a single filter tap;
/// no summation round-off occurs. Error is exactly 0 in f64 representation.
/// Threshold 1×10⁻¹⁵ is conservative.
///
/// Reference: Daubechies (1992), Ten Lectures on Wavelets, p.198 (db4 taps);
///            Mallat (1989), two-channel periodic filter-bank analysis.
pub(crate) fn wavelet_daubechies4_one_level_known_coefficients_fixture(
) -> SuiteResult<PublishedFixtureReport> {
    let plan = DwtPlan::new(4, 1, DiscreteWavelet::Daubechies4)?;
    let coefficients = plan.forward(&[1.0_f64, 0.0, 0.0, 0.0])?;
    let mut actual = Vec::with_capacity(4);
    actual.extend_from_slice(coefficients.approximation());
    actual.extend_from_slice(&coefficients.details()[0]);
    let expected = [
        0.482_962_913_144_534_1,
        0.224_143_868_042_013_4,
        -0.129_409_522_551_260_34,
        0.836_516_303_737_807_9,
    ];
    Ok(published_real_fixture_with_threshold(
        "DWT-Db4",
        "DWT-Db4-1level([1,0,0,0])-coefficients",
        "Daubechies (1992) p.198 db4 taps with periodic QMF analysis: [a0,a1,d0,d1]=[h0,h2,h3,h1]",
        &actual,
        &expected,
        1.0e-15,
    ))
}

/// Daubechies-4 perfect reconstruction: inverse(forward(x)) = x for level 1.
///
/// # Mathematical contract
///
/// For an orthogonal two-channel filter bank (Mallat 1989, Thm.2),
/// synthesis is the inverse of analysis on the same wavelet family:
///   IDWT(DWT(x)) = x.
///
/// This fixture uses db4, N=4, level=1 with a non-trivial signal
/// x=[1,−2,0.5,4].
///
/// f64 periodic convolution + one reconstruction stage gives
/// O(filter_len·ε_f64) ≈ O(4·ε_f64) per output sample; threshold 1×10⁻¹²
/// is >10³× larger than worst-case round-off.
///
/// Reference: Mallat (1989) IEEE TPAMI 11(7), Theorem 2 (perfect reconstruction);
///            Daubechies (1992) db4 orthogonal wavelet construction.
pub(crate) fn wavelet_daubechies4_inverse_perfect_reconstruction_fixture(
) -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, -2.0, 0.5, 4.0];
    let plan = DwtPlan::new(4, 1, DiscreteWavelet::Daubechies4)?;
    let coefficients = plan.forward(&input)?;
    let recovered = plan.inverse(&coefficients)?;
    Ok(published_real_fixture_with_threshold(
        "DWT-Db4",
        "DWT-Db4-inverse-roundtrip-1level(N=4,[1,-2,0.5,4])",
        "Mallat (1989) Thm.2 perfect reconstruction: IDWT(DWT(x))=x for orthogonal db4 filter bank",
        &recovered,
        &input,
        1.0e-12,
    ))
}

/// CWT Ricker impulse response: peak value equals ψ(0) and zero-crossing neighbors are exact zeros.
///
/// # Mathematical contract
///
/// The continuous wavelet transform coefficient at scale a and translation b is:
///   W(a, b) = (1/√a) Σ_n x[n] ψ((n−b)/a)
///
/// For a discrete impulse x = δ_{n₀} (x[n₀]=1, all other samples 0) and scale a=1:
///   W(1, n₀) = ψ(0)
///   W(1, n₀±1) = ψ(±1)
///
/// The normalized Ricker (Mexican hat) wavelet (Daubechies 1992 §2.1 eq.(2.1.4)):
///   ψ(t) = (2/(√3·π^{1/4})) · (1 − t²) · exp(−t²/2)
///
/// Evaluation:
///   ψ(0) = 2/(√3·π^{1/4}) (factor (1−0)·exp(0)=1).
///   ψ(1) = (2/(√3·π^{1/4})) · (1−1) · exp(−0.5) = 0   (exact; (1−t²)=0 when t=±1).
///   ψ(−1) = 0  (ψ is even; same zero factor).
///
/// Threshold 1×10⁻¹⁴: each computed value involves at most one scalar product
/// of the impulse sample (1.0) with a constant. Accumulated FP error is O(ε_f64) ≈ 2×10⁻¹⁶.
///
/// Reference: Daubechies (1992), Ten Lectures on Wavelets, §2.1 eq.(2.1.4);
///            Marr & Hildreth (1980), Proc. R. Soc. Lond. B 207:187–217.
pub(crate) fn cwt_ricker_impulse_peak_value_fixture() -> SuiteResult<PublishedFixtureReport> {
    // Impulse at sample 3 in a 7-sample signal.
    let signal = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let plan = CwtPlan::new(7, vec![1.0], ContinuousWavelet::Ricker)?;
    let coeffs = plan.transform(&signal)?;
    let actual = [
        coeffs.values()[[0, 2]], // W(1, 2) = ψ(1) = 0 exactly
        coeffs.values()[[0, 3]], // W(1, 3) = ψ(0) = 2/(√3·π^{1/4})
        coeffs.values()[[0, 4]], // W(1, 4) = ψ(−1) = 0 exactly
    ];
    let psi0 = 2.0_f64 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25));
    let expected = [0.0_f64, psi0, 0.0_f64];
    Ok(published_real_fixture_with_threshold(
        "CWT-Ricker",
        "CWT-Ricker(δ_{3},N=7,a=1): W(1,2)=0, W(1,3)=ψ(0)=2/(√3·π^¼), W(1,4)=0",
        "Daubechies (1992) Ten Lectures on Wavelets §2.1 eq.(2.1.4): ψ(0)=2/(√3·π^¼); ψ(±1)=0 exact zero crossing",
        &actual,
        &expected,
        1.0e-14,
    ))
}

/// CWT Ricker L² scale-normalization: W(a=2,b=n₀) = ψ(0)/√2.
///
/// # Mathematical contract
///
/// The 1/√a prefactor (L² normalization) from Daubechies (1992) §2.1 is tested
/// directly by comparing impulse-response peaks at two different scales.
///
/// For x = δ_{n₀}, a=2, b=n₀:
///   W(2, n₀) = (1/√2) · ψ((n₀−n₀)/2) = (1/√2) · ψ(0) = ψ(0)/√2
///
/// With ψ(0) = 2/(√3·π^{1/4}) (see fixture 54):
///   W(2, n₀) = 2/(√3·π^{1/4}·√2) = √2/(√3·π^{1/4})
///
/// Threshold 1×10⁻¹³: the only additional computation vs fixture 54 is one
/// multiplication by 1/√2, adding at most one ε_f64 of error. O(2·ε_f64) ≤ 1×10⁻¹³.
///
/// Reference: Daubechies (1992), Ten Lectures on Wavelets, §2.1 L² normalization;
///            Grossmann & Morlet (1984), SIAM J. Math. Anal. 15(4):723–736, eq.(1.3).
pub(crate) fn cwt_ricker_scale_normalization_fixture() -> SuiteResult<PublishedFixtureReport> {
    let signal = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let plan = CwtPlan::new(7, vec![2.0], ContinuousWavelet::Ricker)?;
    let coeffs = plan.transform(&signal)?;
    let actual = [coeffs.values()[[0, 3]]]; // W(2, 3) = ψ(0)/√2
    let psi0_over_sqrt2 =
        2.0_f64 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25) * std::f64::consts::SQRT_2);
    let expected = [psi0_over_sqrt2];
    Ok(published_real_fixture_with_threshold(
        "CWT-Ricker",
        "CWT-Ricker(δ_{3},N=7,a=2): W(2,3)=ψ(0)/√2=√2/(√3·π^¼)",
        "Daubechies (1992) §2.1 L² normalization: W(a,b)=(1/√a)·CWT; Grossmann-Morlet (1984) eq.(1.3)",
        &actual,
        &expected,
        1.0e-13,
    ))
}
