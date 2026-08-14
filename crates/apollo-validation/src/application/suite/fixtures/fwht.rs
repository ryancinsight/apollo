//! Published-reference fixtures for the FWHT transform family.

#![expect(
    clippy::unwrap_used,
    reason = "ratchet APOLLO-UNWRAP-1: pre-existing debt"
)]

use super::super::SuiteResult;
use super::builders::{published_real_fixture, published_real_fixture_with_threshold};
use crate::domain::report::PublishedFixtureReport;
use apollo_fwht::FwhtPlan;
use leto::Array1;

/// FWHT two-point known-value fixture.
///
/// # Mathematical contract
///
/// The 2×2 Hadamard matrix is H₂ = [[1,1],[1,−1]].
/// H₂·[1,1]ᵀ = [2,0]ᵀ (both basis functions evaluate to 1+1 and 1−1 respectively).
/// The FWHT is the unnormalized Hadamard transform.
/// Reference: Hadamard (1893), two-point Walsh-Hadamard matrix definition.
pub(crate) fn fwht_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = FwhtPlan::new(2)?;
    let input = Array1::from(vec![1.0_f64, 1.0]);
    let actual = plan.forward(&input)?;
    let expected = [2.0_f64, 0.0];
    Ok(published_real_fixture(
        "FWHT",
        "FWHT2([1,1])",
        "Hadamard (1893), H_2=[[1,1],[1,-1]], H_2\u{00b7}[1,1]^T=[2,0]^T",
        actual.as_slice().unwrap(),
        &expected,
    ))
}

/// FWHT inverse roundtrip: `inverse(forward(x)) = x` for N=4.
///
/// # Mathematical contract
///
/// Walsh (1923) Am. J. Math. 45 §2: the Walsh-Hadamard matrix W_N satisfies
/// W_N · W_N = N · I_N (involution), so W_N⁻¹ = (1/N)·W_N. The `inverse`
/// method applies W_N then scales by 1/N; composition gives
/// `inverse(forward(x)) = (1/N)·W_N·(W_N·x) = x`. For N=4 f64 butterfly
/// arithmetic, roundtrip error is O(log₂(N)·ε_f64) ≈ 8.9×10⁻¹⁶; threshold 1×10⁻¹⁴.
/// Reference: Walsh (1923) §2: {H_k} is a complete ONS; W_N²=N·I_N.
pub(crate) fn fwht_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, 2.0, 3.0, 4.0];
    let plan = FwhtPlan::new(4)?;
    let signal = Array1::from(input.to_vec());
    let spectrum = plan.forward(&signal)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "FWHT",
        "FWHT-inverse-roundtrip(N=4,[1,2,3,4])",
        "Walsh (1923) Am. J. Math. 45 sec.2: W_N^2=N*I, IFWHT(FWHT(x))=x; Hadamard (1893)",
        recovered.as_slice().unwrap(),
        &input,
        1.0e-14,
    ))
}
