//! Published-reference fixtures for the RADON transform family.

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
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

pub(crate) fn radon_theta0_column_impulse_projection_fixture() -> SuiteResult<PublishedFixtureReport>
{
    // Theorem: parallel-beam Radon projection at θ=0 equals column sums.
    // Image: unit impulse at pixel (row=0, col=0), all other pixels zero.
    //   Column 0 sum = 1.0, column 1 sum = 0.0, column 2 sum = 0.0.
    // Sinogram[θ=0] = [1.0, 0.0, 0.0] (3 detectors, spacing=1.0, column-aligned).
    // Reference: Radon (1917) parallel-beam projection definition (Über die Bestimmung
    //            von Funktionen durch ihre Integralwerte);
    //            Natterer (1986) §I.2: discrete projection at θ=0 equals column sums.
    let mut image = Array2::<f64>::zeros((3, 3));
    image[[0, 0]] = 1.0;
    let plan = RadonPlan::new(3, 3, vec![0.0], 3, 1.0)?;
    let sinogram = plan.forward(&image)?;
    let row: Vec<f64> = sinogram.values().row(0).iter().copied().collect();
    let expected = vec![1.0_f64, 0.0, 0.0];
    Ok(published_real_fixture(
        "Radon",
        "Radon-theta0-column0-impulse-projection(3x3)",
        "Radon (1917): parallel-beam θ=0 projection = column sums; Natterer (1986) §I.2",
        &row,
        &expected,
    ))
}

/// Radon Fourier Slice Theorem at \u{03b8}=0: DFT_1(R_{\u{03b8}=0}f) equals the horizontal
/// slice of the 2D DFT of f (Natterer 1986, Radon 1917).
///
/// # Mathematical contract
///
/// Projection-Slice Theorem (Natterer 1986, §I.2, Theorem 1.1):
///   \u{1d4d5}_1{R_\u{03b8} f}(\u{03c9}) = \u{1d4d5}_2{f}(\u{03c9}·cos\u{03b8}, \u{03c9}·sin\u{03b8})
///
/// For \u{03b8}=0 (parallel-beam at θ=0 = column sums):
///   p_0[n] = Σ_m f[m,n]   (projection = sum over rows at each column detector n)
///   DFT_1(p_0)[k] = Σ_n p_0[n]·exp(-2πi·k·n/N)
///                 = Σ_n (Σ_m f[m,n])·exp(-2πi·k·n/N)
///                 = F_2[0,k]  (horizontal slice of 2D DFT)
///
/// For image f=[[1,2],[3,4]], N=M=2:
///   p_0 = [1+3, 2+4] = [4, 6]  (column sums)
///   DFT_1([4,6]): P[0] = 4+6 = 10,  P[1] = 4+6·exp(-i\u{03c0}) = 4-6 = -2
///   2D DFT check: F_2[0,0]=1+2+3+4=10 ✓,  F_2[0,1]=1-2+3-4=-2 ✓
///
/// All DFT factors are exp(-2πi·k·n/N) ∈ {1, exp(-i\u{03c0})} = {1, -1};
/// computation is exact in f64; accumulated floating-point error = 0.
/// Threshold 1×10⁻¹² is conservative.
///
/// Reference: Natterer (1986) The Mathematics of Computerized Tomography §I.2,
///            Theorem 1.1 (Projection-Slice Theorem); Radon (1917) original
///            parallel-beam projection definition.
pub(crate) fn radon_fourier_slice_theorem_theta0_fixture() -> SuiteResult<PublishedFixtureReport> {
    let image = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0])
        .expect("(2,2) shape is valid for 4 elements");
    let plan = RadonPlan::new(2, 2, vec![0.0_f64], 2, 1.0)?;
    let sinogram = plan.forward(&image)?;
    let projection = sinogram.values().row(0).to_owned();
    let dft_of_projection = fft_1d_array(&projection);
    let expected = [Complex64::new(10.0, 0.0), Complex64::new(-2.0, 0.0)];
    Ok(published_complex_fixture(
        "Radon",
        "DFT_1(R_\u{03b8}=0([[1,2],[3,4]]))-vs-2D-DFT-slice,N=2",
        "Natterer (1986) §I.2 Thm 1.1 Projection-Slice Theorem: DFT_1(R_\u{03b8}f)=F_2{f}(\u{03c9}cos\u{03b8},\u{03c9}sin\u{03b8}); Radon (1917)",
        dft_of_projection.iter(),
        expected.iter(),
    ))
}
