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

/// Iterates over mutable `chunk_size`-element chunks of `data`, passing
/// `(index, chunk)` to `f` through the selected dispatch provider.
#[inline]
pub fn for_each_chunk_mut_enumerated<P, T, F>(data: &mut [T], chunk_size: usize, f: F)
where
    P: moirai::ExecutionPolicy,
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    moirai::for_each_chunk_mut_enumerated_with::<P, _, _>(data, chunk_size, f);
}

#[cfg(test)]
mod tests {
    use super::{for_each_chunk_mut_enumerated, RadixCompositePolicy};
    use moirai::ExecutionPolicy;

    #[test]
    fn workload_threshold_selects_dispatch_mode() {
        assert!(RadixCompositePolicy::parallelize(32_768));
        assert!(!RadixCompositePolicy::parallelize(32_767));
    }

    #[test]
    fn sequential_and_parallel_dispatch_preserve_chunk_index_semantics() {
        let mut sequential = vec![0usize; 16];
        let mut parallel = vec![0usize; 16];

        for_each_chunk_mut_enumerated::<moirai::Sequential, _, _>(
            &mut sequential,
            4,
            |i, chunk| {
                chunk.fill(i);
            },
        );
        for_each_chunk_mut_enumerated::<moirai::Parallel, _, _>(&mut parallel, 4, |i, chunk| {
            chunk.fill(i);
        });

        assert_eq!(sequential, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
        assert_eq!(parallel, sequential);
    }
}
