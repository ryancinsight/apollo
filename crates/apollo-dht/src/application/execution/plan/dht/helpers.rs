//! Thread-local scratch pools and interop/validation helpers for Discrete Hartley Transform.

use crate::domain::contracts::error::{DhtError, DhtResult};
use apollo_fft::PrecisionProfile;
use mnemosyne::scratch::ScratchPool;

thread_local! {
    #[cfg_attr(
        windows,
        expect(
            clippy::missing_const_for_thread_local,
            reason = "false positive: the initializer is already a const block"
        )
    )]
    pub(crate) static LANE_IN_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    #[cfg_attr(
        windows,
        expect(
            clippy::missing_const_for_thread_local,
            reason = "false positive: the initializer is already a const block"
        )
    )]
    pub(crate) static LANE_OUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    #[cfg_attr(
        windows,
        expect(
            clippy::missing_const_for_thread_local,
            reason = "false positive: the initializer is already a const block"
        )
    )]
    pub(crate) static TYPED_INPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    #[cfg_attr(
        windows,
        expect(
            clippy::missing_const_for_thread_local,
            reason = "false positive: the initializer is already a const block"
        )
    )]
    pub(crate) static TYPED_OUTPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

#[inline]
pub(crate) fn validate_profile(
    actual: PrecisionProfile,
    expected: PrecisionProfile,
) -> DhtResult<()> {
    if actual.matches_storage_and_compute(expected) {
        Ok(())
    } else {
        Err(DhtError::PrecisionMismatch)
    }
}
