//! Published-reference fixtures for the GFT transform family.

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
use leto::Array2 as LetoArray2;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// GFT K₂ path graph Laplacian eigenvalues are {0, 2}.
///
/// # Mathematical contract
///
/// For the 2-vertex path graph with adjacency A=[[0,1],[1,0]], the combinatorial
/// Laplacian L = D − A = [[1,−1],[−1,1]]. Its characteristic polynomial is
/// det(L − λI) = (1−λ)² − 1 = λ² − 2λ = λ(λ−2), giving eigenvalues {0, 2}.
/// The `spectral_basis` kernel sorts eigenvalues in ascending order, so the
/// GFT plan returns eigenvalues [0.0, 2.0]. This is a sign-independent,
/// numerically deterministic published result.
/// Reference: Shuman et al. (2013), IEEE Signal Processing Magazine, graph Fourier basis definition.
pub(crate) fn gft_path_graph_forward_fixture() -> SuiteResult<PublishedFixtureReport> {
    let adjacency = LetoArray2::from_shape_vec([2, 2], vec![0.0_f64, 1.0, 1.0, 0.0])?;
    let plan = GftPlan::from_adjacency(adjacency.view())?;
    let eigenvalues = plan.eigenvalues().to_vec();
    let expected = [0.0_f64, 2.0];
    Ok(published_real_fixture(
        "GFT",
        "GFT-K2-path-eigenvalues",
        "Shuman et al. (2013), K\u{2082} path graph Laplacian L=[[1,-1],[-1,1]]: eigenvalues {0,2} from det(L-\u{03bb}I)=\u{03bb}(\u{03bb}-2)",
        &eigenvalues,
        &expected,
    ))
}

/// GFT K₂ path graph inverse roundtrip: inverse(forward(s)) = s.
///
/// # Mathematical contract
///
/// The graph Fourier transform diagonalises the Laplacian via its real symmetric
/// eigenbasis U: GFT(s) = Uᵀs and GFT⁻¹(ŝ) = Uŝ (Sandryhaila & Moura 2013,
/// ICASSP, §II).  Since U is orthonormal (UUᵀ = I), the composition gives
/// GFT⁻¹(GFT(s)) = UUᵀs = s exactly.  For the K₂ path graph with 2×2 Laplacian,
/// eigendecomposition is exact in f64; roundtrip error is bounded by O(N·ε_f64)
/// ≈ 4.4×10⁻¹⁶; threshold is 1×10⁻¹².
/// Reference: Sandryhaila and Moura (2013), ICASSP: GFT via Laplacian eigendecomposition.
pub(crate) fn gft_path_graph_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let adjacency = LetoArray2::from_shape_vec([2, 2], vec![0.0_f64, 1.0, 1.0, 0.0])?;
    let plan = GftPlan::from_adjacency(adjacency.view())?;
    let signal = Array1::from_vec(vec![3.0_f64, -1.0]);
    let spectrum = plan.forward(&signal)?;
    let recovered = plan.inverse(&spectrum)?;
    let expected = [3.0_f64, -1.0];
    Ok(published_real_fixture_with_threshold(
        "GFT",
        "GFT-K2-inverse-roundtrip([3,-1])",
        "Sandryhaila and Moura (2013) ICASSP §II: GFT\u{207b}\u{00b9}(GFT(s))=s via orthonormal Laplacian eigenbasis U; K\u{2082} path graph",
        recovered.as_slice().unwrap(),
        &expected,
        1.0e-12,
    ))
}
