//! Thread-local scratch pools and interop/validation helpers for Discrete Hartley Transform.

use mnemosyne::scratch::ScratchPool;

thread_local! {
    pub(crate) static LANE_IN_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static LANE_OUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Release this module's two thread-local scratch pools on the calling thread.
pub fn release_workspace_scratch() {
    LANE_IN_SCRATCH.with(|pool| pool.release());
    LANE_OUT_SCRATCH.with(|pool| pool.release());
}
