//! Published-reference fixtures for the MELLIN transform family.

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

pub(crate) fn mellin_constant_function_first_moment_fixture() -> SuiteResult<PublishedFixtureReport>
{
    // Mellin moment M(s) = ∫_a^b f(r) r^{s-1} dr for f(r)=1, s=1, a=1, b=3.
    // M(1) = ∫_1^3 1·r^0 dr = ∫_1^3 1 dr = b - a = 2.0.
    // Trapezoid rule on 3 equidistant points r=[1,2,3], h=1:
    //   (h/2)·[f(1)·1 + 2·f(2)·1 + f(3)·1] = (1/2)·[1+2+1] = 2.0 — exact.
    // Reference: Mellin (1897) transform definition M(s)=∫_a^b f(r)r^{s-1}dr;
    //            Titchmarsh (1937) §1.1: M(1)=b-a=2 for f(r)=1 on [1,3].
    let plan = MellinPlan::new(3, 1.0, 3.0)?;
    let signal = [1.0_f64, 1.0, 1.0];
    let moment = plan.moment(&signal, 1.0, 3.0, 1.0)?;
    let actual = [moment];
    let expected = [2.0_f64];
    Ok(published_real_fixture(
        "Mellin",
        "Mellin-constant-function-first-moment([1,3],s=1)",
        "Mellin (1897) transform definition; Titchmarsh (1937) §1.1: M(1)=b-a=2 for f(r)=1 on [1,3]",
        &actual,
        &expected,
    ))
}

/// Mellin inverse spectrum roundtrip for constant signal on N=32 grid.
///
/// # Mathematical contract
///
/// For a constant signal f(r) = c on [a, b], the log-resampled values are
/// identically c, so the DFT of the log-samples has only a DC component F[0] = N·c
/// and F[k] = 0 for k > 0.  The IDFT recovers the constant log-domain signal, and
/// exp-resampling of a constant is c at every output point.  Accumulation error in
/// f64 DFT+IDFT for N=32 is bounded by O(log N)·ε_f64 ≈ 5·2.2×10⁻¹⁶ ≈ 1.1×10⁻¹⁵;
/// the fixture threshold is 1×10⁻¹⁰ (five orders of margin over the dominant
/// interpolation residual; constant function makes interpolation error zero).
/// Reference: Mellin (1896) transform definition; Titchmarsh (1937) §1.1.
pub(crate) fn mellin_inverse_spectrum_constant_roundtrip_fixture(
) -> SuiteResult<PublishedFixtureReport> {
    // MELLIN_INVERSE_ROUNDTRIP_LIMIT: log-domain DFT/IDFT accumulation for N=32
    // plus linear interpolation residual.  Constant signal makes interpolation
    // error zero; f64 DFT error is O(log N)·ε_f64 << 1e-10.
    const MELLIN_INVERSE_ROUNDTRIP_LIMIT: f64 = 1.0e-10;
    let n = 32usize;
    let min_scale = 1.0_f64;
    let max_scale = 4.0_f64;
    let plan = MellinPlan::new(n, min_scale, max_scale)?;
    let signal: Vec<f64> = vec![2.0_f64; n];
    let spectrum = plan.forward_spectrum(&signal, min_scale, max_scale)?;
    let mut recovered = vec![0.0_f64; n];
    plan.inverse_spectrum(&spectrum, min_scale, max_scale, &mut recovered)?;
    let expected: Vec<f64> = vec![2.0_f64; n];
    Ok(published_real_fixture_with_threshold(
        "Mellin",
        "Mellin-inverse-constant-roundtrip(N=32,[1,4],c=2)",
        "Mellin (1896); Titchmarsh (1937) §1.1: constant f(r)=c → DC spectrum → IDFT+exp-resample recovers c",
        &recovered,
        &expected,
        MELLIN_INVERSE_ROUNDTRIP_LIMIT,
    ))
}
