//! Published-reference fixtures for the NUFFT transform family.

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

/// NUFFT Type-1 1D with a single source at x=0 and value 1+0i, N=4, dx=π/2 (L=2π).
///
/// # Mathematical contract
/// F[k] = Σ_j f[j]·exp(-2πi·k_signed·x_j/L). With x_j=0:
///   angle = -2π·k_signed·0/L = 0 for every k.
///   exp(0) = 1+0i exactly in IEEE 754 (cos(0)=1, sin(0)=0 are exact).
/// Therefore F[k] = 1+0i for all k=0..3, with zero floating-point error.
/// Reference: "NUFFT Type-1 definition: F[k] = Σ_j f[j]·exp(-2πi·k·x_j/L); at x_j=0, F[k]=1 for all k"
pub(crate) fn nufft_impulse_at_origin_fixture() -> SuiteResult<PublishedFixtureReport> {
    let domain = UniformDomain1D::new(4, std::f64::consts::FRAC_PI_2)?;
    let actual = nufft_type1_1d(&[0.0], &[Complex64::new(1.0, 0.0)], domain);
    let expected = [Complex64::new(1.0, 0.0); 4];
    Ok(published_complex_fixture(
        "NUFFT",
        "NUFFT-Type1-1D(x=[0], f=[1+0i], N=4)",
        "NUFFT Type-1 definition: F[k] = \u{03a3}_j f[j]\u{00b7}exp(-2\u{03c0}i\u{00b7}k\u{00b7}x_j/L); at x_j=0, F[k]=1 for all k",
        actual.iter(),
        expected.iter(),
    ))
}

/// NUFFT Type-1 1D with a single source at x=L/4 and value 1+0i, N=4, dx=π/2 (L=2π).
///
/// # Mathematical contract
/// F[k] = Σ_j f[j]·exp(-2πi·k_signed·x_j/L). With x_j=L/4 and f[j]=1:
///   angle = -2π·k_signed·(L/4)/L = -π·k_signed/2.
///   k_signed sequence for N=4: k=0→0, k=1→1, k=2→2, k=3→-1.
///   F[0] = exp(0) = 1+0i (exact)
///   F[1] = exp(-πi/2) = cos(-π/2)+i·sin(-π/2) ≈ 0-i  (|error| < 7e-17)
///   F[2] = exp(-πi)   = cos(-π)+i·sin(-π)     ≈ -1+0i (|error| < 2e-16)
///   F[3] = exp(+πi/2) = cos(+π/2)+i·sin(+π/2) ≈ 0+i  (|error| < 7e-17)
/// All errors are well within the 1×10⁻¹² published-fixture threshold.
/// Reference: "NUFFT Type-1 definition: F[k]=exp(-2πi·k_signed·x₀/L) for unit source at x₀=L/4 (Dutt and Rokhlin 1993)"
pub(crate) fn nufft_quarter_period_phase_fixture() -> SuiteResult<PublishedFixtureReport> {
    let domain = UniformDomain1D::new(4, std::f64::consts::FRAC_PI_2)?;
    let x0 = std::f64::consts::FRAC_PI_2; // L/4 = (4·π/2)/4 = π/2
    let actual = nufft_type1_1d(&[x0], &[Complex64::new(1.0, 0.0)], domain);
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    Ok(published_complex_fixture(
        "NUFFT",
        "NUFFT-Type1-1D(x=[L/4], f=[1+0i], N=4)",
        "NUFFT Type-1 definition: F[k]=exp(-2\u{03c0}i\u{00b7}k_signed\u{00b7}x\u{2080}/L) for unit source at x\u{2080}=L/4; F=[1,-i,-1,i] (Dutt and Rokhlin 1993)",
        actual.iter(),
        expected.iter(),
    ))
}

/// NUFFT Type-1/Type-2 adjoint inner-product identity (Dutt-Rokhlin 1993).
///
/// # Mathematical contract
///
/// Type-1 operator A and Type-2 operator A* are defined as:
///   (A·c)[k] = Σ_j c_j · exp(-2πi·k·x_j/L),  k = 0, …, N-1
///   (A*·f)[j] = Σ_k f_k · exp(+2πi·k·x_j/L)
///
/// They satisfy the adjoint identity (proof by index swap):
///   Re(⟨A·c, f⟩) = Re(⟨c, A*·f⟩)
///
/// Analytically derived values for N=2, dx=0.5, L=1.0,
/// positions=[0.0, 0.5], c=[1+0i, 2+0i], f=[3+0i, 4+0i]:
///   All exp factors are exp(-2πi·k·x_j/L) ∈ {1, exp(-πi)} = {1, -1}.
///   Type-1: F[0] = 1·1 + 2·1 = 3;  F[1] = 1·1 + 2·(-1) = -1
///   Type-2: G[0] = 3·1 + 4·1 = 7;  G[1] = 3·1 + 4·(-1) = -1
///   LHS = Re(conj(3)·3 + conj(-1)·4) = 9 - 4 = 5  (exact integer)
///   RHS = Re(conj(1)·7 + conj(2)·(-1)) = 7 - 2 = 5  (exact integer)
///
/// All arithmetic is exact in f64 (factors ∈ {1, −1}); accumulated FP error = 0.
/// Threshold 1×10⁻¹² is conservative; actual error is identically 0.
///
/// Reference: Dutt & Rokhlin (1993) SIAM J. Sci. Comput. 14(6) 1368–1393:
///            adjoint identity eq. (1.8); Greengard & Lee (2004) §2 proof.
pub(crate) fn nufft_type1_type2_adjoint_inner_product_fixture(
) -> SuiteResult<PublishedFixtureReport> {
    let domain = UniformDomain1D::new(2, 0.5)?;
    let positions = vec![0.0_f64, 0.5];
    let c = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let f_arr = Array1::from(vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)]);
    let capital_f = nufft_type1_1d(&positions, &c, domain);
    let g = nufft_type2_1d(&f_arr, &positions, domain);
    let lhs: f64 = capital_f
        .iter()
        .zip(f_arr.iter())
        .map(|(fk, dk)| (fk.conj() * dk).re)
        .sum();
    let rhs: f64 = c
        .iter()
        .zip(g.iter())
        .map(|(cj, gj)| (cj.conj() * gj).re)
        .sum();
    Ok(published_real_fixture_with_threshold(
        "NUFFT",
        "Re(<A\u{00b7}c,f>)=Re(<c,A*\u{00b7}f>),N=2,pos=[0,0.5],c=[1,2],f=[3,4]",
        "Dutt & Rokhlin (1993) SIAM J. Sci. Comput. 14(6): NUFFT Type-1/Type-2 adjoint identity (1.8); Greengard-Lee (2004) \u{00a7}2",
        &[lhs, rhs],
        &[5.0_f64, 5.0],
        1.0e-12,
    ))
}
