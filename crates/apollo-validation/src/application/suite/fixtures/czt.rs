//! Published-reference fixtures for the CZT transform family.

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

/// CZT with DFT parameters equals the DFT for the unit impulse [1,0,0,0].
///
/// # Mathematical contract
///
/// When A=1 and W=exp(−2πi/N), the CZT spiral contour collapses to the unit
/// circle at the N-th roots of unity, recovering the DFT exactly.
/// For input [1+0i,0,0,0], DFT([1,0,0,0])=[1,1,1,1].
/// Therefore CZT(N=4,M=4,A=1,W=exp(−2πi/4))([1,0,0,0])=[1,1,1,1].
/// Reference: Rabiner, Schafer and Rader (1969), CZT spiral-collapse theorem.
pub(crate) fn czt_unit_impulse_is_dft_fixture() -> SuiteResult<PublishedFixtureReport> {
    let n = 4usize;
    let a = Complex64::new(1.0, 0.0);
    let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
    let plan = CztPlan::new(n, n, a, w)?;
    let input = Array1::from(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let actual = plan.forward(&input)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "CZT",
        "CZT4(A=1,W=exp(-2\u{03c0}i/4),[1,0,0,0])",
        "Rabiner, Schafer and Rader (1969), CZT with A=1 and W=exp(-2\u{03c0}i/N) equals DFT; DFT([1,0,0,0])=[1,1,1,1]",
        actual.iter(),
        expected.iter(),
    ))
}

/// CZT inverse Vandermonde roundtrip: inverse(forward(x)) == x for N=4, A=1, W=exp(−2πi/4).
///
/// # Mathematical contract
///
/// The CZT with A=1 and W=exp(−2πi/N) evaluates the DFT at the N-th roots of unity
/// (Rabiner-Schafer-Rader 1969, spiral-collapse theorem).  The inverse solves the
/// resulting Vandermonde system V·x̂ = X via the Björck-Pereyra algorithm (1970),
/// which is backward-stable for N=4 DFT nodes.  Theorem: for any x,
/// CZT⁻¹(CZT(x)) = x in exact arithmetic.  f64 arithmetic gives ‖x − x̂‖∞ < 1 ulp ≈ 2.2×10⁻¹⁶
/// for N=4; the fixture threshold is 1×10⁻¹² (five orders of margin).
/// Reference: Rabiner, Schafer and Rader (1969); Björck and Pereyra (1970), SIAM J. Numer. Anal. 7(6).
pub(crate) fn czt_inverse_vandermonde_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let n = 4usize;
    let a = Complex64::new(1.0, 0.0);
    let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
    let plan = CztPlan::new(n, n, a, w)?;
    // Non-trivial input: unit impulse → DFT = [1,1,1,1]; invert back to [1,0,0,0].
    let input = Array1::from(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "CZT",
        "CZT4-inverse-Vandermonde-roundtrip(A=1,W=exp(-2\u{03c0}i/4),[1,0,0,0])",
        "Rabiner-Schafer-Rader (1969) spiral-collapse; Björck-Pereyra (1970) Vandermonde solve: CZT\u{207b}\u{00b9}(CZT(x))=x",
        recovered.iter(),
        expected.iter(),
    ))
}

/// CZT off-unit-circle Z-transform evaluation: A=2 (real, |A|=2>1), W=exp(-πi).
///
/// # Mathematical contract
///
/// The Chirp Z-Transform evaluates the one-sided Z-transform at spiral points
/// z_k = A·W^{-k} (Rabiner, Schafer & Rader 1969, §II):
///   X[k] = Σ_{n=0}^{N-1} x[n] · A^{-n} · W^{nk},  k=0..M-1
///
/// For N=2, M=2, A=2, W=exp(-πi)=-1, x=[1,1]:
///   z_0 = A·W^0 = 2  (real axis, |z_0|=2 > 1, off the unit circle)
///   z_1 = A·W^{-1} = 2·(-1) = -2  (real axis, |z_1|=2 > 1)
///
///   Z{x}(z) = x[0]·z^{-0} + x[1]·z^{-1} = 1 + z^{-1}
///   X[0] = Z{x}(2)  = 1 + 2^{-1}     = 3/2 = 1.5  (exact dyadic rational)
///   X[1] = Z{x}(-2) = 1 + (-2)^{-1}  = 1/2 = 0.5  (exact dyadic rational)
///
/// Direct check via definition:
///   X[0] = 1·A^0·W^0 + 1·A^{-1}·W^0 = 1 + 0.5           = 1.5 ✓
///   X[1] = 1·A^0·W^0 + 1·A^{-1}·W^1 = 1 + 0.5·(-1) = 0.5 ✓
///
/// 3/2 and 1/2 are exactly representable as IEEE 754 f64 (dyadic fractions);
/// accumulated floating-point error is identically 0. Threshold 1×10^{-12}
/// is conservative (actual error = 0 in exact arithmetic).
///
/// Reference: Rabiner, L.R., Schafer, R.W. & Rader, C.M. (1969).
///            "The Chirp z-Transform Algorithm."
///            IEEE Trans. Audio Electroacoustics 17(2), 86–92. §II.
pub(crate) fn czt_off_unit_circle_z_transform_fixture() -> SuiteResult<PublishedFixtureReport> {
    let a = Complex64::new(2.0, 0.0);
    let w = Complex64::from_polar(1.0, -std::f64::consts::PI);
    let plan = CztPlan::new(2, 2, a, w)?;
    let input = Array1::from(vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]);
    let actual = plan.forward(&input)?;
    let expected = [Complex64::new(1.5, 0.0), Complex64::new(0.5, 0.0)];
    Ok(published_complex_fixture(
        "CZT",
        "CZT2(A=2,W=exp(-πi),[1,1]): Z{x}(2)=1.5,Z{x}(-2)=0.5",
        "Rabiner, Schafer & Rader (1969) IEEE TAE 17(2) §II: CZT at z_k=A·W^{-k}; A=2 evaluates Z-transform off unit circle at z={2,-2}",
        actual.iter(),
        expected.iter(),
    ))
}
