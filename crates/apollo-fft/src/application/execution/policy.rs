//! Execution policy dispatch for zero-cost parallelism selection.
//!
//! Delegates directly to moirai's data-parallel primitives, eliminating
//! the prior redundant `ExecutionPolicy` trait / `SyncPolicy` / `ParallelPolicy`
//! wrapper layer (DRY + SSOT violation with moirai's own `ExecutionPolicy`).
//! `ChunkDispatch` keeps the Apollo-side workload threshold explicit without
//! leaking boolean blindness into provider call sites.

/// Selection policy for composite radix execution.
pub struct RadixCompositePolicy;

impl moirai::ExecutionPolicy for RadixCompositePolicy {
    #[inline]
    fn parallelize(len: usize) -> bool {
        len >= crate::application::execution::kernel::tuning::RADIX_PARALLEL_CHUNK_THRESHOLD
    }
}

#[cfg(test)]
mod tests {
    use super::RadixCompositePolicy;
    use moirai::ExecutionPolicy;

    #[test]
    fn workload_threshold_selects_dispatch_mode() {
        assert!(RadixCompositePolicy::parallelize(32_768));
        assert!(!RadixCompositePolicy::parallelize(32_767));
    }
}
