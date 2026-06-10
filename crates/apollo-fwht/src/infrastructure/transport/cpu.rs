use crate::application::execution::plan::fwht::dimension_1d::FwhtPlan;
use crate::domain::contracts::error::FwhtError;
use ndarray::Array1;

/// Forward FWHT convenience wrapper.
///
/// Creates a plan from the signal length and calls forward.
/// Fails if the signal length is zero or not a power of two.
pub fn fwht(signal: &Array1<f64>) -> Result<Array1<f64>, FwhtError> {
    FwhtPlan::new(signal.len())?.forward(signal)
}

/// Forward FWHT convenience wrapper for a Leto view.
pub fn fwht_leto(
    signal: leto::ArrayView1<'_, f64>,
) -> Result<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>, FwhtError> {
    FwhtPlan::new(signal.shape()[0])?.forward_leto(signal)
}

/// Inverse FWHT convenience wrapper.
///
/// Creates a plan from the spectrum length and calls inverse.
/// Fails if the spectrum length is zero or not a power of two.
pub fn ifwht(spectrum: &Array1<f64>) -> Result<Array1<f64>, FwhtError> {
    FwhtPlan::new(spectrum.len())?.inverse(spectrum)
}

/// Inverse FWHT convenience wrapper for a Leto view.
pub fn ifwht_leto(
    spectrum: leto::ArrayView1<'_, f64>,
) -> Result<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>, FwhtError> {
    FwhtPlan::new(spectrum.shape()[0])?.inverse_leto(spectrum)
}
