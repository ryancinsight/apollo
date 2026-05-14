//! Frequency bin utilities for FFT outputs.
//!
//! ## Mathematical contract
//!
//! `fftfreq(n, d)` returns the discrete Fourier frequency bin centers for a transform of length `n`
//! with sample spacing `d` (seconds per sample). The frequency for bin `k` is:
//!
//! - `k / (n·d)` for `k = 0, 1, …, ⌊(n−1)/2⌋` (non-negative half)
//! - `(k − n) / (n·d)` for `k = ⌊(n−1)/2⌋+1, …, n−1` (negative half)
//!
//! This is the numpy-compatible convention: for even n, bin n/2 is negative (−1/(2d)).
//!
//! `rfftfreq(n, d)` returns the n/2+1 non-negative frequency bins for a real-input FFT:
//! - `k / (n·d)` for `k = 0, 1, …, n/2`
//! - Output length is `n/2 + 1`.

/// Frequency bin centers for a full complex DFT of length `n` with sample spacing `d`.
///
/// Implements numpy-compatible `fftfreq(n, d)`:
/// - Bin `k < (n+1)/2`: `k / (n·d)` (non-negative half)
/// - Bin `k ≥ (n+1)/2`: `(k − n) / (n·d)` (negative half)
///
/// For `n = 0` returns an empty vector.
/// For `d = 0.0` all finite outputs are undefined; no error is raised, matching numpy behavior.
#[must_use]
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let nd = n as f64 * d;
    let half = (n + 1) / 2; // positive bins: 0..half
    let mut bins = Vec::with_capacity(n);
    for k in 0..half {
        bins.push(k as f64 / nd);
    }
    for k in half..n {
        bins.push((k as i64 - n as i64) as f64 / nd);
    }
    bins
}

/// Frequency bin centers for a real-input FFT of length `n` with sample spacing `d`.
///
/// Implements numpy-compatible `rfftfreq(n, d)`:
/// - Returns bins `0, 1, …, n/2` (all non-negative), length `n/2 + 1`.
/// - Bin `k`: `k / (n·d)`.
///
/// For `n = 0` returns `vec![0.0]` (the single DC bin).
/// For `d = 0.0` behavior is undefined; no error is raised, matching numpy behavior.
#[must_use]
pub fn rfftfreq(n: usize, d: f64) -> Vec<f64> {
    if n == 0 {
        return vec![0.0];
    }
    let nd = n as f64 * d;
    let len = n / 2 + 1;
    let mut bins = Vec::with_capacity(len);
    for k in 0..len {
        bins.push(k as f64 / nd);
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// fftfreq(8, 1.0) = [0, 1/8, 2/8, 3/8, -4/8, -3/8, -2/8, -1/8].
    /// Reference: numpy.fft.fftfreq(8, 1.0).
    #[test]
    fn fftfreq_n8_d1_matches_numpy_reference() {
        let bins = fftfreq(8, 1.0);
        let expected = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125];
        assert_eq!(bins.len(), 8);
        for (got, exp) in bins.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(got, exp, epsilon = f64::EPSILON * 4.0);
        }
    }

    /// fftfreq(7, 1.0): odd-n, positive half = 0..4, negative = 4..7.
    /// numpy reference: [0, 1/7, 2/7, 3/7, -3/7, -2/7, -1/7].
    #[test]
    fn fftfreq_n7_odd_matches_numpy_reference() {
        let bins = fftfreq(7, 1.0);
        let expected = [
            0.0_f64,
            1.0 / 7.0,
            2.0 / 7.0,
            3.0 / 7.0,
            -3.0 / 7.0,
            -2.0 / 7.0,
            -1.0 / 7.0,
        ];
        assert_eq!(bins.len(), 7);
        for (got, exp) in bins.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(got, exp, epsilon = 1.0e-14);
        }
    }

    /// fftfreq(1, 1.0) = [0.0].
    #[test]
    fn fftfreq_n1() {
        let bins = fftfreq(1, 1.0);
        assert_eq!(bins, [0.0]);
    }

    /// fftfreq(0, 1.0) = [].
    #[test]
    fn fftfreq_n0_returns_empty() {
        let bins = fftfreq(0, 1.0);
        assert!(bins.is_empty());
    }

    /// fftfreq scales linearly with 1/d.
    #[test]
    fn fftfreq_scales_with_d() {
        let bins_1hz = fftfreq(8, 1.0);
        let bins_2hz = fftfreq(8, 2.0);
        for (a, b) in bins_1hz.iter().zip(bins_2hz.iter()) {
            assert_abs_diff_eq!(a, &(b * 2.0), epsilon = f64::EPSILON * 4.0);
        }
    }

    /// rfftfreq(8, 1.0) = [0, 1/8, 2/8, 3/8, 4/8], length 5.
    /// Reference: numpy.fft.rfftfreq(8, 1.0).
    #[test]
    fn rfftfreq_n8_d1_matches_numpy_reference() {
        let bins = rfftfreq(8, 1.0);
        let expected = [0.0, 0.125, 0.25, 0.375, 0.5];
        assert_eq!(bins.len(), 5);
        for (got, exp) in bins.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(got, exp, epsilon = f64::EPSILON * 4.0);
        }
    }

    /// rfftfreq always returns non-negative values.
    #[test]
    fn rfftfreq_all_non_negative() {
        for n in [1_usize, 2, 4, 7, 8, 16] {
            for &f in rfftfreq(n, 1.0).iter() {
                assert!(f >= 0.0, "rfftfreq({n}, 1.0) has negative bin {f}");
            }
        }
    }

    /// rfftfreq length is n/2+1.
    #[test]
    fn rfftfreq_length_is_n_over_2_plus_1() {
        for n in [1_usize, 2, 4, 7, 8, 16] {
            assert_eq!(rfftfreq(n, 1.0).len(), n / 2 + 1);
        }
    }

    /// rfftfreq(0, 1.0) = [0.0] (single DC bin).
    #[test]
    fn rfftfreq_n0_returns_dc() {
        let bins = rfftfreq(0, 1.0);
        assert_eq!(bins, [0.0]);
    }
}
