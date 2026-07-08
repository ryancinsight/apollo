use crate::application::execution::plan::stft::dimension_1d::StftPlan;
use crate::domain::contracts::error::StftResult;
use eunomia::Complex64;
use leto::Array1;

/// Forward STFT convenience wrapper.
///
/// Constructs a `StftPlan` with the given parameters and calls `forward`.
pub fn stft(
    signal: &Array1<f64>,
    frame_len: usize,
    hop_len: usize,
) -> StftResult<Array1<Complex64>> {
    StftPlan::new(frame_len, hop_len)?.forward(signal)
}

/// Forward STFT convenience wrapper for Leto signal views.
pub fn stft_leto(
    signal: leto::ArrayView1<'_, f64>,
    frame_len: usize,
    hop_len: usize,
) -> StftResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
    StftPlan::new(frame_len, hop_len)?.forward_leto(signal)
}

/// Inverse STFT convenience wrapper.
///
/// Constructs a `StftPlan` with the given parameters and calls `inverse`.
pub fn istft(
    spectrum: &Array1<Complex64>,
    frame_len: usize,
    hop_len: usize,
    signal_len: usize,
) -> StftResult<Array1<f64>> {
    StftPlan::new(frame_len, hop_len)?.inverse(spectrum, signal_len)
}

/// Inverse STFT convenience wrapper for Leto spectrum views.
pub fn istft_leto(
    spectrum: leto::ArrayView1<'_, Complex64>,
    frame_len: usize,
    hop_len: usize,
    signal_len: usize,
) -> StftResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
    StftPlan::new(frame_len, hop_len)?.inverse_leto(spectrum, signal_len)
}
