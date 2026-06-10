//! Published-reference fixtures for the NTT transform family.

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

/// NTT of the unit impulse [1,0,0,0] with N=4, modulus=998244353, primitive root=3.
///
/// # Mathematical contract
/// By the NTT definition (Pollard 1971): F[k] = Σ_{n=0}^{N-1} f[n]·ω^{nk} mod p.
/// With f[0]=1 and f[1..3]=0, every term except n=0 vanishes, giving F[k] = ω^0 = 1 for all k.
/// Reference: "NTT definition, Pollard (1971): F[k] = Σ f[n]·ω^{nk} mod p, impulse response F[k]=1"
pub(crate) fn ntt_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![1u64, 0, 0, 0]);
    let actual = ntt(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64, 1.0, 1.0, 1.0];
    Ok(published_real_fixture(
        "NTT",
        "NTT4([1,0,0,0])",
        "NTT definition, Pollard (1971): F[k] = \u{03a3} f[n]\u{00b7}\u{03c9}^{nk} mod p, impulse response F[k]=1",
        &actual_f64,
        &expected,
    ))
}

/// NTT of the constant-one vector [1,1,1,1] with N=4, modulus=998244353, primitive root=3.
///
/// # Mathematical contract
/// F[0] = Σ_{n=0}^3 1 = 4 (mod p). For k≠0: F[k] = Σ_{n=0}^3 ω^{nk}.
/// Since ω is a primitive N-th root of unity and ω^k≠1 for k=1,2,3, the geometric series
/// Σ_{n=0}^{N-1} (ω^k)^n = (ω^{Nk}-1)/(ω^k-1) = 0 mod p, because ω^N ≡ 1 (mod p).
/// Reference: "NTT DFT-of-constant theorem: F[0]=N, F[k≠0]=0 for constant input (Pollard 1971)"
pub(crate) fn ntt_constant_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![1u64, 1, 1, 1]);
    let actual = ntt(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [4.0_f64, 0.0, 0.0, 0.0];
    Ok(published_real_fixture(
        "NTT",
        "NTT4([1,1,1,1])",
        "NTT DFT-of-constant theorem: F[0]=N, F[k\u{2260}0]=0 for constant input (Pollard 1971)",
        &actual_f64,
        &expected,
    ))
}

/// NTT of the unit impulse [1,0,0,0,0,0,0,0] with N=8, modulus=998244353, primitive root=3.
///
/// # Mathematical contract
/// By the NTT definition (Pollard 1971): F[k] = Σ_{n=0}^{N-1} f[n]·ω^{nk} mod p.
/// With f[0]=1 and f[1..7]=0, every term except n=0 vanishes, giving F[k] = ω^0 = 1 for all k.
/// This is the same impulse theorem as N=4, generalized to N=8.
/// Reference: "NTT definition, Pollard (1971): F[k] = Σ f[n]·ω^{nk} mod p, impulse response F[k]=1 (N=8)"
pub(crate) fn ntt_n8_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![1u64, 0, 0, 0, 0, 0, 0, 0]);
    let actual = ntt(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64; 8];
    Ok(published_real_fixture(
        "NTT",
        "NTT8([1,0,0,0,0,0,0,0])",
        "NTT definition, Pollard (1971): F[k] = \u{03a3} f[n]\u{00b7}\u{03c9}^{nk} mod p, impulse response F[k]=1 (N=8)",
        &actual_f64,
        &expected,
    ))
}

/// NTT convolution theorem: NTT^{-1}(NTT(a) ⊙ NTT(b)) = a ★ b for polynomial product.
///
/// # Mathematical contract
/// By the NTT Convolution Theorem (Pollard 1971): for a=[1,2,0,0] and b=[3,4,0,0],
/// the cyclic convolution a★b equals the coefficients of (1+2x)(3+4x) = 3+10x+8x², giving c=[3,10,8,0].
/// All values satisfy 3,10,8 ≪ p = 998244353 so modular reduction is trivial.
/// The fixture verifies INTT(NTT(a) ⊙ NTT(b)) against the analytically derived polynomial product.
/// Reference: "NTT Convolution Theorem (Pollard 1971): INTT(NTT(a)⊙NTT(b)) = a★b mod p"
pub(crate) fn ntt_polynomial_convolution_fixture() -> SuiteResult<PublishedFixtureReport> {
    let p = DEFAULT_MODULUS;
    let plan = NttPlan::new(4)?;
    let a = Array1::from_vec(vec![1u64, 2, 0, 0]);
    let b = Array1::from_vec(vec![3u64, 4, 0, 0]);
    let fa = plan.forward(&a)?;
    let fb = plan.forward(&b)?;
    let fc: Vec<u64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&x, &y)| ((x as u128 * y as u128) % p as u128) as u64)
        .collect();
    let fc_arr = Array1::from(fc);
    let c = intt(&fc_arr)?;
    let actual_f64: Vec<f64> = c.iter().map(|&v| v as f64).collect();
    // (1+2x)(3+4x) = 3+10x+8x^2 → [3,10,8,0]
    let expected = [3.0_f64, 10.0, 8.0, 0.0];
    Ok(published_real_fixture(
        "NTT",
        "INTT(NTT([1,2,0,0])\u{2299}NTT([3,4,0,0]))",
        "NTT Convolution Theorem (Pollard 1971): INTT(NTT(a)\u{2299}NTT(b)) = a\u{2605}b mod p; (1+2x)(3+4x)=3+10x+8x\u{00b2}",
        &actual_f64,
        &expected,
    ))
}

/// NTT of the unit impulse [1, 0, 0, ..., 0] with N=16, modulus=998244353.
///
/// # Mathematical contract
/// By the NTT impulse theorem (Pollard 1971): for x[0]=1 and x[n]=0 for n≥1,
///   F[k] = Σ_{n=0}^{N-1} x[n] · ω^{nk} = ω^{0·k} = 1  for all k = 0..15.
/// This extends the N=4 and N=8 impulse fixtures to N=16, confirming that the
/// twiddle precomputation and butterfly structure are correct at this size.
/// Reference: "NTT impulse theorem, Pollard (1971): F[k]=1 ∀k for impulse input"
pub(crate) fn ntt_n16_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let mut input_vec = vec![0u64; 16];
    input_vec[0] = 1;
    let input = Array1::from_vec(input_vec);
    let plan = NttPlan::new(16)?;
    let actual = plan.forward(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64; 16];
    Ok(published_real_fixture(
        "NTT",
        "NTT16([1,0,...,0])",
        "NTT impulse theorem, Pollard (1971): F[k]=1 for all k; impulse input, N=16",
        &actual_f64,
        &expected,
    ))
}

/// NTT convolution theorem for N=16: degree-3 times degree-1 polynomial product.
///
/// # Mathematical contract
/// By the NTT Convolution Theorem (Pollard 1971):
///   INTT(NTT(a) ⊙ NTT(b)) = cyclic convolution a ★ b  mod p
/// For a = [1,2,3,4,0,...,0] and b = [2,1,0,...,0] (N=16):
///   (1 + 2x + 3x² + 4x³)(2 + x) = 2 + 5x + 8x² + 11x³ + 4x⁴
/// All coefficients (2, 5, 8, 11, 4) are ≪ p = 998244353, so modular reduction
/// is trivial and the result equals the integer polynomial product exactly.
/// Reference: "NTT Convolution Theorem (Pollard 1971): INTT(NTT(a)⊙NTT(b))=a★b; N=16"
pub(crate) fn ntt_n16_polynomial_product_fixture() -> SuiteResult<PublishedFixtureReport> {
    let p = DEFAULT_MODULUS;
    let plan = NttPlan::new(16)?;
    let mut a_vec = vec![0u64; 16];
    a_vec[..4].copy_from_slice(&[1u64, 2, 3, 4]);
    let mut b_vec = vec![0u64; 16];
    b_vec[..2].copy_from_slice(&[2u64, 1]);
    let a = Array1::from_vec(a_vec);
    let b = Array1::from_vec(b_vec);
    let fa = plan.forward(&a)?;
    let fb = plan.forward(&b)?;
    let fc: Vec<u64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&x, &y)| ((x as u128 * y as u128) % p as u128) as u64)
        .collect();
    let c = plan.inverse(&Array1::from_vec(fc))?;
    let actual_f64: Vec<f64> = c.iter().map(|&v| v as f64).collect();
    // (1+2x+3x²+4x³)(2+x) = 2 + 5x + 8x² + 11x³ + 4x⁴, all higher coefficients 0
    let mut expected = [0.0_f64; 16];
    expected[..5].copy_from_slice(&[2.0, 5.0, 8.0, 11.0, 4.0]);
    Ok(published_real_fixture(
        "NTT",
        "INTT(NTT([1,2,3,4,0\u{2026}])\u{2299}NTT([2,1,0\u{2026}]))",
        "NTT Convolution Theorem (Pollard 1971): (1+2x+3x\u{00b2}+4x\u{00b3})(2+x)=2+5x+8x\u{00b2}+11x\u{00b3}+4x\u{2074}; N=16",
        &actual_f64,
        &expected,
    ))
}

/// NTT 4-point inverse roundtrip: INTT(NTT(x)) = x.
///
/// # Mathematical contract
///
/// Let p = 998244353 (= 119·2²³+1), N=4, g=3, ω = g^{(p-1)/N} mod p.
/// By Pollard (1971) Theorem 1: NTT is a bijection on (ℤ/pℤ)^N with exact integer
/// inverse INTT(X)[n] = (1/N) Σ_k X[k] · ω^{-nk} mod p.
/// For x=[1,2,3,4]: NTT[k] = Σ_{n=0}^3 x[n]·ω^{nk} mod p.
/// INTT(NTT(x)) = x holds exactly in ℤ/pℤ arithmetic.
/// Converting to f64: all values ≤ 4 ≪ 2^53, so representation is exact.
/// Floating-point error in f64 comparison: 0 (exact integer reconstruction).
/// Threshold 1×10⁻¹² covers f64 conversion round-trip noise.
/// Reference: Pollard (1971) Math. Proc. Cambridge Phil. Soc. 70(3): NTT inversion theorem.
pub(crate) fn ntt_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = NttPlan::new(4)?;
    let input = Array1::from_vec(vec![1u64, 2, 3, 4]);
    let spectrum = plan.forward(&input)?;
    let recovered = intt(&spectrum)?;
    let recovered_f64: Vec<f64> = recovered.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64, 2.0, 3.0, 4.0];
    Ok(published_real_fixture(
        "NTT",
        "INTT(NTT([1,2,3,4]),N=4)",
        "Pollard (1971) Math. Proc. Cambridge Phil. Soc. 70(3): NTT inversion theorem; INTT(NTT(x))=x in \u{2124}/p\u{2124} for p=998244353",
        &recovered_f64,
        &expected,
    ))
}
