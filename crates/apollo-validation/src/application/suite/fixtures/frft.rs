//! Published-reference fixtures for the FRFT transform family.

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

/// Unitary FrFT of order 2 is the reversal operator.
///
/// # Mathematical contract
///
/// By Candan et al. (2000), Theorem 3: DFrFT₂(x)[k] = x[N−1−k] for all x.
/// The palindrome-structured Grünbaum eigenvectors diagonalize the DFT; the
/// fractional phase exp(−i·2·k·π/2) = (−1)^k combines with V·diag((−1)^k)·Vᵀ
/// to produce the exact index-reversal operator.
/// Input [1+0i, 2+0i, 3+0i, 4+0i], N=4, order=2.
/// Expected output: [4+0i, 3+0i, 2+0i, 1+0i].
pub(crate) fn frft_unitary_order2_reversal_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let plan = UnitaryFrftPlan::new(4, 2.0)?;
    let actual = plan.forward(&input)?;
    let expected = [
        Complex64::new(4.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "UnitaryFrFT",
        "UnitaryFrFT(order=2,[1,2,3,4])",
        "Candan et al. (2000), Theorem 3: DFrFT\u{2082} = reversal operator; UnitaryFrFT(order=2,[1,2,3,4])=[4,3,2,1]",
        actual.iter(),
        expected.iter(),
    ))
}

/// Unitary FrFT inverse roundtrip: FrFT(−α)(FrFT(α)(x)) = x for α=0.5.
///
/// # Mathematical contract
///
/// By the FrFT additivity theorem (Namias 1980, J. IMA 25(3) §2):
/// F^α ∘ F^β = F^{α+β} for all real orders α, β.  Setting β = −α gives
/// F^{−α}(F^α(x)) = F^0(x) = x (identity at order 0).  For α=0.5, N=4:
/// the unitary plan's `inverse` method applies `forward` with negated order,
/// so the composition reduces exactly to order 0, which is the identity.
/// f64 roundtrip error via Grünbaum eigendecomposition for N=4 is bounded by
/// O(N²·ε_f64) ≈ 16×2.2×10⁻¹⁶ ≈ 3.5×10⁻¹⁵ per transform, O(7×10⁻¹⁵) total;
/// threshold is 1×10⁻¹² (three orders of margin).
/// Reference: Namias (1980), J. Inst. Math. Appl. 25(3); Candan et al. (2000) §II.
pub(crate) fn frft_inverse_roundtrip_order_half_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let plan = UnitaryFrftPlan::new(4, 0.5)?;
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "UnitaryFrFT",
        "UnitaryFrFT-inverse-roundtrip(alpha=0.5,N=4,[1,2,3,4])",
        "Namias (1980) J.IMA 25(3) sec.2 FrFT additivity: F^{-alpha}(F^alpha(x))=x; Candan et al. (2000) sec.II",
        recovered.iter(),
        expected.iter(),
    ))
}

/// Unitary FrFT order \u03b1=4 is the identity on any input.
///
/// # Mathematical contract
///
/// Candan, Kutay & Ozaktas (2000) \u00a7II, Corollary (periodicity):
///   DFrFT_a(x) = V \u00b7 diag(exp(\u2212i\u00b7a\u00b7k\u00b7\u03c0/2)) \u00b7 V\u1d40 \u00b7 x
/// For a=4: exp(\u22124\u00b7k\u00b7\u03c0\u00b7i/2) = exp(\u22122\u03c0ki) = 1 for every integer k.
/// Therefore DFrFT_4 = V \u00b7 I \u00b7 V\u1d40 = I (since V is orthogonal, V\u1d40V = I).
///
/// This holds for any N and any input x, independent of the eigenvector
/// ordering in V and without requiring a=1 to equal the DFT.
///
/// For N=4, two O(N\u00b2) matrix\u2013vector products: accumulated FP error
/// \u2248 2\u00b7N\u00b7\u03b5_f64 = 2\u00b74\u00b72.2\u00d710\u207b\u00b9\u2076 \u2248 1.8\u00d710\u207b\u00b9\u2075; threshold 1\u00d710\u207b\u00b9\u00b2 (\u00d770 margin).
///
/// Reference: Candan, \u00c7., Kutay, M.A., & Ozaktas, H.M. (2000).
///            \"The Discrete Fractional Fourier Transform.\"
///            IEEE Trans. Signal Process. 48(5), 1329\u20131337. \u00a7II Corollary.
pub(crate) fn frft_order4_identity_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let plan = UnitaryFrftPlan::new(4, 4.0)?;
    let output = plan.forward(&input)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "UnitaryFrFT",
        "UnitaryFrFT-order4-identity(N=4,[1,2,3,4])",
        "Candan et al. (2000) IEEE TSP 48(5) §II Corollary: DFrFT_4=I; exp(-i·4kπ/2)=exp(-2πki)=1; V·I·V^T=I",
        output.iter(),
        expected.iter(),
    ))
}
