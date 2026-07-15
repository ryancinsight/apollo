//! Thread-local scratch pools and interop/validation helpers for Discrete Hartley Transform.

use crate::domain::contracts::error::{DhtError, DhtResult};
use apollo_fft::PrecisionProfile;
use mnemosyne::scratch::ScratchPool;

thread_local! {
    pub(crate) static LANE_IN_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static LANE_OUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static TYPED_INPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static TYPED_OUTPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

#[cfg(feature = "wgpu")]
pub(crate) fn leto_view1_cow<'a, T: Copy>(
    view: &leto::ArrayView1<'a, T>,
) -> std::borrow::Cow<'a, [T]> {
    apollo_fft::application::utilities::leto_interop::view1_cow(view)
}

#[inline]
pub(crate) fn validate_profile(
    actual: PrecisionProfile,
    expected: PrecisionProfile,
) -> DhtResult<()> {
    if apollo_fft::application::utilities::leto_interop::profile_matches(actual, expected) {
        Ok(())
    } else {
        Err(DhtError::PrecisionMismatch)
    }
}
