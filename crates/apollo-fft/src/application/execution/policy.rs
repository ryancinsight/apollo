//! Execution policy dispatch for zero-cost parallelism selection.
//!
//! Delegates directly to moirai's data-parallel primitives, eliminating
//! the prior redundant `ExecutionPolicy` trait / `SyncPolicy` / `ParallelPolicy`
//! wrapper layer (DRY + SSOT violation with moirai's own `ExecutionPolicy`).
//! The boolean `parallel` flag selects between sequential and parallel
//! execution at each call site, matching the existing `(src_is_data, use_parallel)`
//! dispatch pattern in `radix_composite::core`.

/// Iterates over mutable `chunk_size`-element chunks of `data`, passing
/// `(index, chunk)` to `f`. When `parallel` is true, dispatches through
/// moirai's work-stealing scheduler; otherwise runs sequentially.
#[inline]
pub fn for_each_chunk_mut_enumerated<T, F>(
    data: &mut [T],
    chunk_size: usize,
    parallel: bool,
    f: F,
) where
    T: Send + Sync,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    if parallel {
        moirai::for_each_chunk_mut_enumerated_with::<moirai::Parallel, _, _>(data, chunk_size, f);
    } else {
        data.chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    }
}
