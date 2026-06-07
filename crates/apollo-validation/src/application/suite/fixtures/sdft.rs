//! Published-reference fixtures for the SDFT transform family.

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

/// SDFT bin 0 for the unit impulse [1, 0, 0, 0].
///
/// # Mathematical contract
///
/// DFT bin k=0: F[0] = Σ_{n=0}^{N-1} x[n] = 1 + 0 + 0 + 0 = 1 for the unit impulse.
/// The SDFT direct-bins path computes the standard DFT for the given window exactly.
/// Reference: Jacobsen and Lyons (2003), Sliding DFT, IEEE Signal Processing Magazine.
pub(crate) fn sdft_bin_zero_unit_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, 0.0, 0.0, 0.0];
    let plan = SdftPlan::new(4, 4)?;
    let bins = plan.direct_bins(&input)?;
    let actual_arr = [bins[0]];
    let expected_arr = [Complex64::new(1.0, 0.0)];
    Ok(published_complex_fixture(
        "SDFT",
        "SDFT-bin0-impulse([1,0,0,0])",
        "Jacobsen and Lyons (2003), SDFT bin-0 at position 0 equals DFT bin-0 = sum(x) = 1 for unit impulse",
        actual_arr.iter(),
        expected_arr.iter(),
    ))
}

/// SDFT sliding-update recurrence agreement: after N updates on zero state
/// fed with the unit impulse [1,0,0,0], all 4 tracked bins equal 1+0i.
///
/// # Mathematical contract
///
/// Jacobsen & Lyons (2003) §2, eq.(2): the sliding DFT recurrence
///   X_k[n] = (X_k[n-1] + x[n] - x[n-N]) \u00b7 exp(2\u03c0i\u00b7k/N)
/// produces exactly the DFT of the current window x[n-N+1..=n] at every step.
///
/// Analytical derivation for N=4, zero_state, input [1,0,0,0]:
///   Twiddles (exp(2\u03c0ik/4)): k=0 \u21a6 1, k=1 \u21a6 i, k=2 \u21a6 -1, k=3 \u21a6 -i.
///   Update 0 (x_new=1): bins = [1,i,-1,-i] (window=[0,0,0,1])
///     DFT([0,0,0,1])[k] = exp(-2\u03c0i\u00b73k/4): k=0\u21a61, k=1\u21a6i, k=2\u21a6-1, k=3\u21a6-i \u2713
///   Update 1 (x_new=0): bins\u2190bins*twiddles = [1,-1,1,-1] (window=[0,0,1,0])
///     DFT([0,0,1,0])[k] = exp(-2\u03c0i\u00b72k/4): k=0\u21a61, k=1\u21a6-1, k=2\u21a61, k=3\u21a6-1 \u2713
///   Update 2 (x_new=0): bins\u2190bins*twiddles = [1,-i,-1,i] (window=[0,1,0,0])
///     DFT([0,1,0,0])[k] = exp(-2\u03c0i\u00b7k/4): k=0\u21a61, k=1\u21a6-i, k=2\u21a6-1, k=3\u21a6i \u2713
///   Update 3 (x_new=0): bins\u2190bins*twiddles = [1,1,1,1] (window=[1,0,0,0])
///     DFT([1,0,0,0])[k] = 1 for all k \u2713
///
/// All factors \u2208{1,i,-1,-i}; result is exact integers; accumulated FP error = 0.
/// Threshold 1\u00d710\u207b\u00b9\u00b2 is conservative.
///
/// Reference: Jacobsen, E. & Lyons, R. (2003). \"The Sliding DFT.\"
///            IEEE Signal Processing Magazine 20(2), 74\u201379. Section 2, eq.(2).
pub(crate) fn sdft_sliding_recurrence_unit_impulse_all_bins_fixture(
) -> SuiteResult<PublishedFixtureReport> {
    let plan = SdftPlan::new(4, 4)?;
    let mut state = plan.zero_state();
    for &sample in &[1.0_f64, 0.0, 0.0, 0.0] {
        state.update(sample);
    }
    let bins = state.bins().to_vec();
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "SDFT",
        "SDFT-sliding-recurrence-impulse(N=4,[1,0,0,0]),all-bins=1",
        "Jacobsen & Lyons (2003) IEEE SPM 20(2) §2 eq.(2): sliding-update recurrence X_k[n]=(X_k[n-1]+x[n]-x[n-N])·exp(2πik/N); DFT([1,0,0,0])=[1,1,1,1]",
        bins.iter(),
        expected.iter(),
    ))
}
