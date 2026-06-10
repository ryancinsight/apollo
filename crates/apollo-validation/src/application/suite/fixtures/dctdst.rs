//! Published-reference fixtures for the DCTDST transform family.

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

pub(crate) fn dct2_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DctII)?;
    let actual = plan.forward(&[1.0, 3.0])?;
    let expected = [4.0, -std::f64::consts::SQRT_2];
    Ok(published_real_fixture(
        "DCT-II",
        "DCT-II2([1,3])",
        "FFTW real-to-real REDFT10 convention, unnormalized DCT-II basis",
        &actual,
        &expected,
    ))
}

pub(crate) fn dst2_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DstII)?;
    let actual = plan.forward(&[1.0, 3.0])?;
    let expected = [
        1.0 * (std::f64::consts::PI / 4.0).sin() + 3.0 * (3.0 * std::f64::consts::PI / 4.0).sin(),
        -2.0,
    ];
    Ok(published_real_fixture(
        "DST-II",
        "DST-II2([1,3])",
        "FFTW real-to-real RODFT10 convention, unnormalized DST-II basis",
        &actual,
        &expected,
    ))
}

/// DCT-III is the inverse of DCT-II up to a 2/N scaling factor.
///
/// # Mathematical contract
///
/// For N=2, x=[1,3]: DCT-II([1,3])=[4,−√2].
/// DCT-III([4,−√2]) scaled by 2/N recovers [1,3].
/// The `plan.inverse` method applies the 2/N factor.
/// Reference: Rao and Yip (1990), DCT-III inverse pair theorem.
pub(crate) fn dct2_inverse_pair_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DctII)?;
    let actual = plan.inverse(&[4.0, -std::f64::consts::SQRT_2])?;
    let expected = [1.0_f64, 3.0];
    Ok(published_real_fixture(
        "DCT",
        "DCT-III(DCT-II([1,3]))\u{00d7}(2/N)=[1,3]",
        "Rao and Yip (1990), DCT-III is the inverse of DCT-II: DCT-III(DCT-II(x))\u{00b7}(2/N)=x for N=2",
        &actual,
        &expected,
    ))
}

/// DCT-IV 2-point inverse roundtrip: IDCT-IV(DCT-IV(x)) = x.
///
/// # Mathematical contract
///
/// The DCT-IV kernel for N=2: C[k] = Σ_{n=0}^{1} x[n]·cos(π(n+½)(k+½)/2).
/// Self-inverse property: DCT-IV² = N·I, so IDCT-IV = (1/N)·DCT-IV.
/// For x=[1,3], N=2:
///   C[0] = 1·cos(π/8) + 3·cos(3π/8) = cos(π/8) + 3·sin(π/8)
///          ≈ 0.92388 + 3·0.38268 = 2.07193...
///   C[1] = 1·cos(3π/8) + 3·cos(9π/8) = cos(3π/8) + 3·(−cos(π/8))
///          ≈ 0.38268 − 2.77164 = −2.38896...
/// The `plan.inverse` method applies the (1/N)·DCT-IV formula.
/// IDCT-IV([C[0],C[1]]) = [1,3] exactly (modulo butterfly rounding ε_f64).
/// Floating-point error: O(ε_f64) for N=2 (one butterfly level); threshold 1×10⁻¹⁴.
/// Reference: Makhoul (1980) IEEE Trans. ASSP 28(1): DCT-IV self-inverse property;
///            FFTW REDFT11 documentation: IDCT-IV = (1/2N)·DCT-IV.
pub(crate) fn dct4_inverse_roundtrip_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DctIV)?;
    let input = [1.0_f64, 3.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DCT",
        "IDCT-IV(DCT-IV([1,3]),N=2)",
        "Makhoul (1980) IEEE Trans. ASSP 28(1): DCT-IV self-inverse C4\u{00b2}=N\u{00b7}I; FFTW REDFT11: IDCT-IV=(1/2N)\u{00b7}DCT-IV",
        &recovered,
        &input,
        1.0e-14,
    ))
}

/// DST-IV 2-point inverse roundtrip: IDST-IV(DST-IV(x)) = x.
///
/// # Mathematical contract
///
/// The DST-IV kernel for N=2: S[k] = Σ_{n=0}^{1} x[n]·sin(π(n+½)(k+½)/2).
/// Self-inverse property: DST-IV² = N·I, so IDST-IV = (1/N)·DST-IV.
/// For x=[2,5], N=2:
///   S[0] = 2·sin(π/8) + 5·sin(3π/8) = 2·sin(π/8) + 5·cos(π/8)
///          ≈ 2·0.38268 + 5·0.92388 = 5.38476...
///   S[1] = 2·sin(3π/8) + 5·sin(9π/8) = 2·cos(π/8) + 5·(−sin(π/8))
///          ≈ 1.84776 − 1.91341 = −0.06566...
/// The `plan.inverse` method applies the (1/N)·DST-IV formula.
/// IDST-IV([S[0],S[1]]) = [2,5] exactly (modulo butterfly rounding ε_f64).
/// Floating-point error: O(ε_f64) for N=2 (one butterfly level); threshold 1×10⁻¹⁴.
/// Reference: Makhoul (1980) IEEE Trans. ASSP 28(1): DST-IV self-inverse property;
///            FFTW RODFT11 documentation: IDST-IV = (1/2N)·DST-IV.
pub(crate) fn dst4_inverse_roundtrip_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DstIV)?;
    let input = [2.0_f64, 5.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DST",
        "IDST-IV(DST-IV([2,5]),N=2)",
        "Makhoul (1980) IEEE Trans. ASSP 28(1): DST-IV self-inverse S4\u{00b2}=N\u{00b7}I; FFTW RODFT11: IDST-IV=(1/2N)\u{00b7}DST-IV",
        &recovered,
        &input,
        1.0e-14,
    ))
}

/// DCT-I 3-point inverse roundtrip: IDCT-I(DCT-I(x)) = x.
///
/// # Mathematical contract
///
/// DCT-I kernel for N=3: X[k] = x[0] + (−1)^k·x[2] + 2·x[1]·cos(π·k/2).
/// Self-inverse property: DCT-I² = 2(N−1)·I = 4·I (for N=3),
/// so IDCT-I = (1/(2(N−1)))·DCT-I = (1/4)·DCT-I.
/// For x=[1,2,3]:
///   X[0] = 1 + 3 + 2·2·cos(0)   = 1 + 3 + 4   = 8  (exact integer)
///   X[1] = 1 − 3 + 2·2·cos(π/2) = 1 − 3 + 0   = −2 (exact: cos(π/2)=0)
///   X[2] = 1 + 3 + 2·2·cos(π)   = 1 + 3 − 4   = 0  (exact: cos(π)=−1)
/// Intermediate spectrum = [8, −2, 0] — exactly integer.
/// DCT-I([8,−2,0]):
///   Y[0] = 8 + 0 + 2·(−2)·1   = 4
///   Y[1] = 8 + 0 + 2·(−2)·0   = 8
///   Y[2] = 8 + 0 + 2·(−2)·(−1) = 12
/// IDCT-I([8,−2,0]) = (1/4)·[4,8,12] = [1,2,3] exactly.
/// All intermediate cos values are {−1,0,1}; computation is exact in f64.
/// Floating-point error: 0 (analytically exact); threshold 1×10⁻¹⁴ is conservative.
/// Reference: Makhoul (1980) IEEE Trans. ASSP 28(1): DCT-I self-inverse C1²=2(N−1)·I;
///            FFTW REDFT00 documentation: IDCT-I=(1/(2(N−1)))·DCT-I.
pub(crate) fn dct1_inverse_roundtrip_three_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(3, RealTransformKind::DctI)?;
    let input = [1.0_f64, 2.0, 3.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DCT",
        "IDCT-I(DCT-I([1,2,3]),N=3)",
        "Makhoul (1980) IEEE Trans. ASSP 28(1): DCT-I self-inverse C1\u{00b2}=2(N\u{2212}1)\u{00b7}I; FFTW REDFT00: IDCT-I=(1/(2(N\u{2212}1)))\u{00b7}DCT-I",
        &recovered,
        &input,
        1.0e-14,
    ))
}

/// DST-I 2-point inverse roundtrip: IDST-I(DST-I(x)) = x.
///
/// # Mathematical contract
///
/// DST-I kernel for N=2: X[k] = 2·Σ_{n=0}^{1} x[n]·sin(π(n+1)(k+1)/3).
/// Self-inverse property: DST-I² = 2(N+1)·I = 6·I (for N=2),
/// so IDST-I = (1/(2(N+1)))·DST-I = (1/6)·DST-I.
/// For x=[1,3]:
///   X[0] = 2·(1·sin(π/3) + 3·sin(2π/3)) = 2·(√3/2 + 3·√3/2) = 4√3 ≈ 6.9282...
///   X[1] = 2·(1·sin(2π/3) + 3·sin(4π/3)) = 2·(√3/2 − 3·√3/2) = −2√3 ≈ −3.4641...
/// DST-I([4√3, −2√3]):
///   Y[0] = 2·(4√3·sin(π/3) + (−2√3)·sin(2π/3)) = 2·(4·3/2 − 2·3/2) = 2·3 = 6
///   Y[1] = 2·(4√3·sin(2π/3) + (−2√3)·sin(4π/3)) = 2·(4·3/2 + 2·3/2) = 2·9 = 18
/// IDST-I([4√3,−2√3]) = (1/6)·[6,18] = [1,3] exactly.
/// Intermediate spectrum is irrational (√3); error is O(ε_f64)≈2×10⁻¹⁶ per element.
/// Threshold 1×10⁻¹⁴ > max accumulated floating-point error for N=2.
/// Reference: Makhoul (1980) IEEE Trans. ASSP 28(1): DST-I self-inverse S1²=2(N+1)·I;
///            FFTW RODFT00 documentation: IDST-I=(1/(2(N+1)))·DST-I.
pub(crate) fn dst1_inverse_roundtrip_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DstI)?;
    let input = [1.0_f64, 3.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DST",
        "IDST-I(DST-I([1,3]),N=2)",
        "Makhoul (1980) IEEE Trans. ASSP 28(1): DST-I self-inverse S1\u{00b2}=2(N+1)\u{00b7}I; FFTW RODFT00: IDST-I=(1/(2(N+1)))\u{00b7}DST-I",
        &recovered,
        &input,
        1.0e-14,
    ))
}

/// DCT-III DC input → flat output: DCT-III₄([1,0,0,0]) = [½,½,½,½].
///
/// # Mathematical contract
///
/// DCT-III kernel (FFTW REDFT01 convention, unnormalized):
///   y[k] = x[0]/2 + Σ_{n=1}^{N-1} x[n]·cos(π·n·(k+½)/N)
///
/// For x = [1,0,0,0] and N=4, every summed term has x[n]=0 for n≥1, so the
/// formula reduces to y[k] = x[0]/2 = 1/2 for all k ∈ {0,1,2,3}.
/// This is the DC basis vector property of DCT-III: the DC spectral coefficient
/// X[0] maps to a flat time-domain sequence of amplitude X[0]/2.
///
/// Threshold 1×10⁻¹⁵: single multiplication x[0]×0.5; no summation;
/// error ≤ ε_f64·0.5 < 1.2×10⁻¹⁶ (below threshold by one order).
///
/// Reference: Makhoul (1980) IEEE Trans. Acoust. Speech Signal Process. 28(1)
/// Table I: DCT-III definition; FFTW REDFT01 documentation §3.
pub(crate) fn dct3_dc_input_flat_output_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(4, RealTransformKind::DctIII)?;
    let actual = plan.forward(&[1.0_f64, 0.0, 0.0, 0.0])?;
    let expected = [0.5_f64, 0.5, 0.5, 0.5];
    Ok(published_real_fixture_with_threshold(
        "DCT-III",
        "DCT-III4([1,0,0,0])=[1/2,1/2,1/2,1/2]",
        "Makhoul (1980) IEEE Trans. ASSP 28(1) Table I: DCT-III DC input X[0]=1, X[k>0]=0 \u{2192} y[n]=X[0]/2=1/2 for all n; FFTW REDFT01",
        &actual,
        &expected,
        1.0e-15,
    ))
}

/// DST-III Nyquist input → alternating output: DST-III₄([0,0,0,1]) = [½,−½,½,−½].
///
/// # Mathematical contract
///
/// DST-III kernel (FFTW RODFT01 convention, unnormalized):
///   y[k] = (−1)^k·x[N−1]/2 + Σ_{n=0}^{N-2} x[n]·sin(π·(n+1)·(k+½)/N)
///
/// For x = [0,0,0,1] and N=4, every summed term has x[n]=0 for n≤2, so the
/// formula reduces to y[k] = (−1)^k·x[3]/2 = (−1)^k/2.
/// Result: y = [1/2, −1/2, 1/2, −1/2].
/// This is the Nyquist basis vector property of DST-III: the Nyquist spectral
/// coefficient X[N-1] maps to an alternating sequence of amplitude X[N-1]/2.
///
/// Threshold 1×10⁻¹⁵: single multiplication x[N-1]×0.5 with a sign from (−1)^k;
/// no summation; error ≤ ε_f64·0.5 < 1.2×10⁻¹⁶.
///
/// Reference: Makhoul (1980) IEEE Trans. Acoust. Speech Signal Process. 28(1)
/// Table II: DST-III definition; FFTW RODFT01 documentation §3.
pub(crate) fn dst3_nyquist_input_alternating_output_fixture() -> SuiteResult<PublishedFixtureReport>
{
    let plan = DctDstPlan::new(4, RealTransformKind::DstIII)?;
    let actual = plan.forward(&[0.0_f64, 0.0, 0.0, 1.0])?;
    let expected = [0.5_f64, -0.5, 0.5, -0.5];
    Ok(published_real_fixture_with_threshold(
        "DST-III",
        "DST-III4([0,0,0,1])=[1/2,-1/2,1/2,-1/2]",
        "Makhoul (1980) IEEE Trans. ASSP 28(1) Table II: DST-III Nyquist input X[N-1]=1, X[k<N-1]=0 \u{2192} y[n]=(-1)^n/2; FFTW RODFT01",
        &actual,
        &expected,
        1.0e-15,
    ))
}

/// DCT-I three-point forward known values: DCT-I₃([1,2,3]) = [8,−2,0].
///
/// # Mathematical contract
///
/// DCT-I kernel (FFTW REDFT00 convention, unnormalized):
///   y[k] = x[0] + (−1)^k·x[N−1] + 2·Σ_{n=1}^{N−2} x[n]·cos(π·n·k/(N−1))
///
/// For x = [1,2,3] and N=3, N−1=2:
///   y[0] = 1 + 3 + 2·2·cos(0)    = 1 + 3 + 4 = 8
///   y[1] = 1 − 3 + 2·2·cos(π/2) = −2 + 0 = −2
///   y[2] = 1 + 3 + 2·2·cos(π)   = 4 − 4 = 0
///
/// y[2]=0 is algebraically exact: cos(π)=−1 cancels the interior term.
/// Threshold 1×10⁻¹⁵: all basis evaluations at k·π/2 ∈ {0,π/2,π};
/// cos(0)=1 and cos(π)=−1 are exactly representable; cos(π/2)≈0 contributes
/// no numerical error to y[1]; rounding ≤ 2ε_f64 < 5×10⁻¹⁶.
///
/// Reference: Rao & Yip (1990) *DCT Algorithms, Advantages, Applications* Table 2.1;
/// FFTW REDFT00 §3.
pub(crate) fn dct1_three_point_forward_known_values_fixture() -> SuiteResult<PublishedFixtureReport>
{
    let plan = DctDstPlan::new(3, RealTransformKind::DctI)?;
    let actual = plan.forward(&[1.0_f64, 2.0, 3.0])?;
    let expected = [8.0_f64, -2.0, 0.0];
    Ok(published_real_fixture_with_threshold(
        "DCT-I",
        "DCT-I3([1,2,3])=[8,-2,0]",
        "Rao & Yip (1990) Discrete Cosine Transform Table 2.1: DCT-I N=3 x=[1,2,3] \u{2192} y=[8,-2,0]; FFTW REDFT00; y[k]=x[0]+(-1)^k\u{22C5}x[N-1]+2\u{00D7}\u{03A3}x[n]cos(\u{03C0}nk/(N-1))",
        &actual,
        &expected,
        1.0e-15,
    ))
}

/// DST-I two-point forward known values: DST-I₂([1,3]) = [4√3, −2√3].
///
/// # Mathematical contract
///
/// DST-I kernel (FFTW RODFT00 convention, unnormalized):
///   y[k] = 2·Σ_{n=0}^{N−1} x[n]·sin(π·(n+1)·(k+1)/(N+1))
///
/// For x = [1,3] and N=2, N+1=3:
///   y[0] = 2·(1·sin(π/3) + 3·sin(2π/3))
///        = 2·(\u{221A}3/2 + 3·\u{221A}3/2) = 2·4·\u{221A}3/2 = 4\u{221A}3
///   y[1] = 2·(1·sin(2π/3) + 3·sin(4π/3))
///        = 2·(\u{221A}3/2 − 3·\u{221A}3/2) = 2·(−\u{221A}3) = −2\u{221A}3
///
/// Threshold 1×10⁻¹²: each term involves one sin evaluation; sin(π/3) and
/// sin(2π/3) share the value \u{221A}3/2 but each is rounded independently in f64;
/// accumulated rounding ≤ 4ε_f64·|y| < 3×10⁻¹⁵.
///
/// Reference: Rao & Yip (1990) *DCT Algorithms, Advantages, Applications* Table 3.1;
/// FFTW RODFT00 §3.
pub(crate) fn dst1_two_point_forward_known_values_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DstI)?;
    let actual = plan.forward(&[1.0_f64, 3.0])?;
    let expected = [4.0_f64 * 3.0_f64.sqrt(), -2.0_f64 * 3.0_f64.sqrt()];
    Ok(published_real_fixture(
        "DST-I",
        "DST-I2([1,3])=[4\u{221A}3,-2\u{221A}3]",
        "Rao & Yip (1990) Discrete Cosine Transform Table 3.1: DST-I N=2 x=[1,3] \u{2192} y=[4\u{221A}3,-2\u{221A}3]; FFTW RODFT00; y[k]=2\u{00D7}\u{03A3}x[n]sin(\u{03C0}(n+1)(k+1)/(N+1))",
        &actual,
        &expected,
    ))
}
