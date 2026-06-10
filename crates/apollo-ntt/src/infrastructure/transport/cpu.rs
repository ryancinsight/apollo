use crate::application::execution::plan::ntt::dimension_1d::NttPlan;
use crate::domain::contracts::error::NttError;
use ndarray::Array1;

/// Forward NTT convenience wrapper; constructs a default-modulus plan and executes forward.
pub fn ntt(input: &Array1<u64>) -> Result<Array1<u64>, NttError> {
    NttPlan::new(input.len())?.forward(input)
}

/// Forward NTT convenience wrapper for a Leto view.
pub fn ntt_leto(
    input: leto::ArrayView1<'_, u64>,
) -> Result<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>, NttError> {
    NttPlan::new(input.shape()[0])?.forward_leto(input)
}

/// Inverse NTT convenience wrapper; constructs a default-modulus plan and executes inverse.
pub fn intt(input: &Array1<u64>) -> Result<Array1<u64>, NttError> {
    NttPlan::new(input.len())?.inverse(input)
}

/// Inverse NTT convenience wrapper for a Leto view.
pub fn intt_leto(
    input: leto::ArrayView1<'_, u64>,
) -> Result<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>, NttError> {
    NttPlan::new(input.shape()[0])?.inverse_leto(input)
}
