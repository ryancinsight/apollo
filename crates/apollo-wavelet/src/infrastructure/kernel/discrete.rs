//! Orthogonal periodic DWT filter-bank kernels.
//!
//! ## Quadrature Mirror Filter (QMF) Identity
//!
//! Theorem: For an orthogonal wavelet with lowpass filter `h[k]`, k = 0…L-1,
//! the highpass decomposition filter is `g[k] = (-1)^k · h[L-1-k]`.
//!
//! Proof: The perfect-reconstruction (PR) condition for a two-channel filter
//! bank requires H(z)·H(z⁻¹) + G(z)·G(z⁻¹) = 2 (Smith-Barnwell condition).
//! The choice G(z) = z^{-(L-1)} · H(-z⁻¹) satisfies this condition when H is
//! linear-phase and orthogonal, and in the time domain reduces to
//! `g[k] = (-1)^k · h[L-1-k]`. This is confirmed for Haar (L=2) and Daubechies4
//! (L=4) coefficients by direct substitution:
//!   Haar: `g[0] = +h[1] = 1/√2`, `g[1] = -h[0] = -1/√2`. ✓
//!   Daubechies4: `g[k] = (-1)^k · h[3-k]` matches the published db4 highpass
//!   coefficients (Daubechies, 1992, p. 198). ✓
//!
//! Corollary: The g array depends only on filter_index ∈ 0…L-1, not on the
//! output index `out`. Precomputing g once before the outer loop eliminates
//! filter_len redundant multiplications per output coefficient.

use crate::domain::metadata::wavelet::DiscreteWavelet;

/// Return low-pass decomposition filter coefficients.
#[must_use]
pub fn lowpass_filter(wavelet: DiscreteWavelet) -> &'static [f64] {
    match wavelet {
        DiscreteWavelet::Haar => &[
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ],
        DiscreteWavelet::Daubechies4 => &[
            0.482_962_913_144_534_1,
            0.836_516_303_737_807_9,
            0.224_143_868_042_013_4,
            -0.129_409_522_551_260_34,
        ],
    }
}

/// Precompute the highpass QMF coefficients g[k] = (-1)^k · h[L-1-k].
///
/// By the QMF theorem (module doc), this array is identical for every output
/// sample index, so computing it once eliminates filter_len redundant
/// multiplications per output coefficient.
#[inline]
fn precompute_highpass(h: &[f64], g: &mut [f64]) {
    debug_assert_eq!(h.len(), g.len());
    let l = h.len();
    for k in 0..l {
        let sign = if k % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
        g[k] = sign * h[l - 1 - k];
    }
}

/// Perform one periodic orthogonal DWT analysis stage without allocations.
///
/// Applies the lowpass filter h and the precomputed highpass QMF g to compute
/// the approximation and detail sub-bands via periodic (wrap-around) convolution
/// followed by downsampling by 2.
///
/// # Contract
/// - `input.len()` must be even and > 0.
/// - `approximation.len() == detail.len() == input.len() / 2`.
pub fn analysis_stage_into(
    input: &[f64],
    wavelet: DiscreteWavelet,
    approximation: &mut [f64],
    detail: &mut [f64],
) {
    let n = input.len();
    debug_assert!(n > 0 && n % 2 == 0);
    debug_assert_eq!(approximation.len(), n / 2);
    debug_assert_eq!(detail.len(), n / 2);

    let h = lowpass_filter(wavelet);
    let filter_len = h.len();

    // Precompute g once — QMF identity (see module doc theorem).
    // Maximum filter length for supported wavelets is 4 (Daubechies4).
    let mut g_buf = [0.0_f64; 8];
    let g = &mut g_buf[..filter_len];
    precompute_highpass(h, g);

    approximation.fill(0.0);
    detail.fill(0.0);

    let half = n / 2;
    for out in 0..half {
        for filter_index in 0..filter_len {
            let input_index = (2 * out + filter_index) % n;
            approximation[out] += h[filter_index] * input[input_index];
            detail[out] += g[filter_index] * input[input_index];
        }
    }
}

/// Perform one periodic orthogonal DWT synthesis stage without allocations.
///
/// Reconstructs the signal from the approximation and detail sub-bands using
/// the lowpass h and highpass QMF g filters, upsampling by 2 and summing.
///
/// # Contract
/// - `approximation.len() == detail.len()`.
/// - `output.len() == 2 * approximation.len()`.
pub fn synthesis_stage_into(
    approximation: &[f64],
    detail: &[f64],
    wavelet: DiscreteWavelet,
    output: &mut [f64],
) {
    debug_assert_eq!(approximation.len(), detail.len());
    let half = approximation.len();
    let n = half * 2;
    debug_assert_eq!(output.len(), n);

    let h = lowpass_filter(wavelet);
    let filter_len = h.len();

    // Precompute g once — QMF identity (see module doc theorem).
    let mut g_buf = [0.0_f64; 8];
    let g = &mut g_buf[..filter_len];
    precompute_highpass(h, g);

    output.fill(0.0);

    for coeff_index in 0..half {
        for filter_index in 0..filter_len {
            let output_index = (2 * coeff_index + filter_index) % n;
            output[output_index] += h[filter_index] * approximation[coeff_index]
                + g[filter_index] * detail[coeff_index];
        }
    }
}
