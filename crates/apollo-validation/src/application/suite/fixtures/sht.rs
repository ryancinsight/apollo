//! Published-reference fixtures for the SHT transform family.

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

pub(crate) fn sht_monopole_y00_coefficient_fixture() -> SuiteResult<PublishedFixtureReport> {
    // f(θ,φ) = Y_0^0(θ,φ) = 1/√(4π) is the spherical harmonic of degree 0.
    // By orthonormality of spherical harmonics Y_l^m:
    //   a_{lm} = ∫_S² f · Y_{lm}^* dΩ = δ_{l,0}δ_{m,0}
    // Therefore a_{00} = 1.0 exactly; all higher-degree coefficients are zero.
    // Gauss-Legendre quadrature with n_lat=12 integrates degree-0 polynomials
    // exactly; the uniform longitude sum with n_lon=25 integrates the constant exactly.
    // Reference: Varshalovich, Moskalev & Khersonskii (1988) §4.1: Y_0^0 = 1/√(4π);
    //            Driscoll & Healy (1994) sampling theorem on S².
    let plan = ShtPlan::new(12, 25, 1)?;
    let constant = 1.0_f64 / (4.0 * std::f64::consts::PI).sqrt();
    let samples = Array2::from_elem(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        constant,
    );
    let coefficients = plan.forward_real(&samples)?;
    let a00 = coefficients.get(0, 0);
    let actual = [a00];
    let expected = [Complex64::new(1.0, 0.0)];
    Ok(published_complex_fixture(
        "SHT",
        "SHT-monopole-Y00-maps-to-unit-coefficient",
        "Varshalovich et al. (1988) §4.1; Driscoll & Healy (1994): Y_0^0=1/sqrt(4π), orthonormality gives a_{00}=1",
        actual.iter(),
        expected.iter(),
    ))
}

/// SHT inverse roundtrip for dipole Y_1^0 (Driscoll-Healy 1994).
///
/// # Mathematical contract
///
/// Driscoll & Healy (1994) Adv. Appl. Math. 15, Theorem 1: for a field
/// band-limited to degree ≤ L, the SHT is exactly invertible when the grid
/// satisfies N_lat > L and N_lon ≥ 2L+1. The dipole Y_1^0(θ,φ) = √(3/4π)·cos θ
/// has degree 1 ≤ lmax=1; plan grid: N_lat=12 > 1, N_lon=25 ≥ 3. Gauss-Legendre
/// weights integrate cos θ products to machine precision; uniform longitude DFT
/// recovers azimuthal modes exactly. Roundtrip error O((L+1)²·ε_f64) < 1×10⁻¹³;
/// threshold 1×10⁻¹⁰ (three orders of margin).
/// Reference: Driscoll & Healy (1994) §2 sampling theorem on S².
pub(crate) fn sht_inverse_roundtrip_y10_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = ShtPlan::new(12, 25, 1)?;
    let n_lat = plan.grid().latitudes();
    let n_lon = plan.grid().longitudes();
    let mut samples_flat = Vec::with_capacity(n_lat * n_lon);
    let mut sample_arr = leto::Array2::<f64>::zeros([n_lat, n_lon]);
    for lat in 0..n_lat {
        let theta = plan.theta(lat);
        let val = (3.0_f64 / (4.0 * std::f64::consts::PI)).sqrt() * theta.cos();
        for lon in 0..n_lon {
            sample_arr[[lat, lon]] = val;
            samples_flat.push(val);
        }
    }
    let coefficients = plan.forward_real(&sample_arr)?;
    let recovered = plan.inverse_real(&coefficients)?;
    let recovered_flat: Vec<f64> = recovered.iter().copied().collect();
    Ok(published_real_fixture_with_threshold(
        "SHT",
        "SHT-Y10-inverse-roundtrip(lmax=1,lat=12,lon=25)",
        "Driscoll & Healy (1994) Adv. Appl. Math. 15 Theorem 1: SHT invertible for band-limited fields; Y_1^0=sqrt(3/4pi)*cos(theta)",
        &recovered_flat,
        &samples_flat,
        1.0e-10,
    ))
}
