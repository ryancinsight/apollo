//! Thread-local scratch pools and interop/validation helpers for 1D CZT.

use crate::domain::contracts::error::CztError;
use apollo_fft::PrecisionProfile;
use eunomia::Complex64;
use mnemosyne::scratch::ScratchPool;
thread_local! {
    pub(crate) static TYPED_INPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    pub(crate) static TYPED_OUTPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    pub(crate) static FORWARD_WORKSPACE_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Run `f` with a thread-local Bluestein forward-convolution workspace sized to `len`.
#[inline]
pub(crate) fn with_forward_workspace<R>(len: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    FORWARD_WORKSPACE_SCRATCH.with(|pool| pool.with_scratch(len, f))
}

/// Return whether a CZT input or output length satisfies the non-zero contract.
#[must_use]
#[inline]
pub fn is_valid_length(n: usize) -> bool {
    n > 0
}

#[inline]
pub(crate) fn validate_profile(
    actual: PrecisionProfile,
    expected: PrecisionProfile,
) -> Result<(), CztError> {
    if actual.matches_storage_and_compute(expected) {
        Ok(())
    } else {
        Err(CztError::PrecisionMismatch)
    }
}

#[inline]
pub(crate) fn with_complex64_workspaces<R>(
    input_len: usize,
    output_len: usize,
    f: impl FnOnce(&mut [Complex64], &mut [Complex64]) -> R,
) -> R {
    TYPED_INPUT64_SCRATCH.with(|in_pool| {
        in_pool.with_scratch(input_len, |input64| {
            TYPED_OUTPUT64_SCRATCH
                .with(|out_pool| out_pool.with_scratch(output_len, |output64| f(input64, output64)))
        })
    })
}

#[cfg(test)]
pub(crate) fn typed_scratch_capacities() -> (usize, usize) {
    TYPED_INPUT64_SCRATCH.with(|in_pool| {
        TYPED_OUTPUT64_SCRATCH.with(|out_pool| (in_pool.capacity(), out_pool.capacity()))
    })
}

#[cfg(test)]
pub(crate) fn forward_workspace_capacity() -> usize {
    FORWARD_WORKSPACE_SCRATCH.with(|pool| pool.capacity())
}
