//! Thread-local scratch pools and interop/validation helpers for Discrete Hartley Transform.

use crate::domain::contracts::error::{DhtError, DhtResult};
use apollo_fft::PrecisionProfile;
use mnemosyne::scratch::ScratchPool;
use leto::{Array2, Array3};
use eunomia::Complex64;

thread_local! {
    pub(crate) static FAST_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    pub(crate) static LANE_IN_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static LANE_OUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static TYPED_INPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static TYPED_OUTPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
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

#[must_use]
#[inline]
pub(crate) fn leto_array2_from_dense(
    output: &Array2<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 2> {
    apollo_fft::application::utilities::leto_interop::try_dense_from_contiguous(output)
        .expect("DHT-owned 2D dense output must be contiguous with matching Leto shape")
}

#[must_use]
#[inline]
pub(crate) fn leto_array3_from_dense(
    output: &Array3<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 3> {
    apollo_fft::application::utilities::leto_interop::try_dense_from_contiguous(output)
        .expect("DHT-owned 3D dense output must be contiguous with matching Leto shape")
}
