//! Thread-local scratch pools and interop/validation helpers for Discrete Hartley Transform.

use mnemosyne::scratch::ScratchPool;

thread_local! {
    pub(crate) static LANE_IN_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static LANE_OUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Release this module's two thread-local lane scratch pools on the calling
/// thread: capacities return to their recorded provisions (zero unless a
/// bounded provision was recorded). Called by the crate-level release;
/// the sixth slice of APOLLO-MEM-SIBLING-THREAD-SCRATCH.
pub fn release_workspace_scratch() {
    LANE_IN_SCRATCH.with(|pool| pool.release());
    LANE_OUT_SCRATCH.with(|pool| pool.release());
}

/// Release every thread-local scratch pool this crate owns on the calling
/// thread: the plan lane pools, the fast-kernel complex pool, and the
/// direct Hartley-row pool. The sixth slice of
/// APOLLO-MEM-SIBLING-THREAD-SCRATCH, following the apollo-fwht,
/// apollo-gft, apollo-dctdst, apollo-czt, and apollo-nufft precedents.
pub fn release_scratch() {
    release_workspace_scratch();
    crate::infrastructure::kernel::fast::release_fast_scratch();
    crate::infrastructure::kernel::direct::release_direct_scratch();
}

#[cfg(test)]
pub(crate) fn lane_scratch_capacities() -> (usize, usize) {
    LANE_IN_SCRATCH
        .with(|in_pool| LANE_OUT_SCRATCH.with(|out_pool| (in_pool.capacity(), out_pool.capacity())))
}
