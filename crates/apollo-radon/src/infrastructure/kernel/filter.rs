//! Projection-domain filters for filtered backprojection.

use apollo_fft::{fft_1d_array, ifft_1d_array};
use leto::Array1;
use eunomia::Complex64;

/// Apply the unwindowed ramp filter `|omega|` to one projection.
#[must_use]
pub fn ramp_filter_projection(projection: &[f64], detector_spacing: f64) -> Vec<f64> {
    let mut output = vec![0.0; projection.len()];
    ramp_filter_projection_into(projection, detector_spacing, &mut output);
    output
}

/// Apply the unwindowed ramp filter into caller-owned storage.
///
/// This avoids per-angle output allocation in filtered backprojection while
/// preserving the same direct DFT mathematical contract as
/// `ramp_filter_projection`.
///
/// # Algorithm
///
/// The ramp filter is applied in the frequency domain:
/// 1. Forward FFT of the projection via `apollo_fft::fft_1d_array` (O(N log N)).
/// 2. Multiply each complex coefficient by `|omega_k|`, the ramp frequency for bin k.
/// 3. Inverse FFT via `apollo_fft::ifft_1d_array`, returning the real-valued filtered
///    projection.
///
/// Using `apollo_fft` as the single source of truth for DFT arithmetic removes the
/// previously duplicated O(N²) private `forward_dft_real` kernel that was also present
/// in `apollo-hilbert`. Both crates now share the same authoritative O(N log N) path.
pub fn ramp_filter_projection_into(projection: &[f64], detector_spacing: f64, output: &mut [f64]) {
    assert_eq!(
        projection.len(),
        output.len(),
        "ramp filter output length mismatch"
    );
    if projection.is_empty() {
        return;
    }

    let arr = Array1::from((projection.iter().copied()).collect::<Vec<_>>());
    let mut spectrum: Vec<Complex64> = fft_1d_array(&arr).to_vec();

    for (k, coefficient) in spectrum.iter_mut().enumerate() {
        *coefficient *= ramp_frequency(k, projection.len(), detector_spacing);
    }

    let spectrum_arr = Array1::from(spectrum);
    let result = ifft_1d_array(&spectrum_arr);

    output
        .iter_mut()
        .zip(result.iter())
        .for_each(|(dst, src)| *dst = *src);
}

fn ramp_frequency(k: usize, len: usize, detector_spacing: f64) -> f64 {
    let signed_bin = if k <= len / 2 {
        k as f64
    } else {
        k as f64 - len as f64
    };
    std::f64::consts::TAU * signed_bin.abs() / (len as f64 * detector_spacing)
}
