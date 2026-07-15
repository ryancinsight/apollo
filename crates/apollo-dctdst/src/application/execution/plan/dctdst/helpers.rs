use crate::domain::contracts::error::{DctDstError, DctDstResult};
use apollo_fft::PrecisionProfile;
pub(crate) fn validate_profile(
    actual: PrecisionProfile,
    expected: PrecisionProfile,
) -> DctDstResult<()> {
    if actual.matches_storage_and_compute(expected) {
        Ok(())
    } else {
        Err(DctDstError::PrecisionMismatch)
    }
}
