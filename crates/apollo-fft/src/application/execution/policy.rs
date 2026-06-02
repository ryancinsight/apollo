//! Execution policy abstractions for zero-cost static dispatch over parallelism regimes.

/// Zero-cost execution policy trait mapping parallelism strategies to compile-time
/// monomorphized variants.
pub trait ExecutionPolicy: Send + Sync + 'static {
    /// Iterates over mutable chunks of `data`, passing `(index, chunk)` to `f`.
    fn for_each_chunk_mut_enumerated<T, F>(data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send + Sync,
        F: Fn(usize, &mut [T]) + Send + Sync;
}

/// Sequential (synchronous) execution policy.
pub struct SyncPolicy;

impl ExecutionPolicy for SyncPolicy {
    #[inline(always)]
    fn for_each_chunk_mut_enumerated<T, F>(data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send + Sync,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        data.chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    }
}

/// Data-parallel execution policy backed by moirai's work-stealing scheduler.
pub struct ParallelPolicy;

impl ExecutionPolicy for ParallelPolicy {
    #[inline(always)]
    fn for_each_chunk_mut_enumerated<T, F>(data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send + Sync,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        moirai::for_each_chunk_mut_enumerated_with::<moirai::Parallel, _, _>(data, chunk_size, f);
    }
}
