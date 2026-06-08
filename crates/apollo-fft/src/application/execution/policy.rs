//! Execution policy dispatch for zero-cost parallelism selection.
//!
//! Delegates directly to moirai's data-parallel primitives, eliminating
//! the prior redundant `ExecutionPolicy` trait / `SyncPolicy` / `ParallelPolicy`
//! wrapper layer (DRY + SSOT violation with moirai's own `ExecutionPolicy`).
//! `ChunkDispatch` keeps the Apollo-side workload threshold explicit without
//! leaking boolean blindness into provider call sites.

/// Chunk execution mode for Apollo-to-Moirai mutable partition dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChunkDispatch {
    /// Execute chunks sequentially on the caller thread.
    Sequential,
    /// Execute chunks through Moirai's parallel work-stealing scheduler.
    Parallel,
}

impl ChunkDispatch {
    /// Selects parallel dispatch when both total workload and per-chunk work
    /// meet their measured thresholds.
    #[must_use]
    #[inline]
    pub const fn for_workload(
        total_len: usize,
        chunk_size: usize,
        min_total_len: usize,
        min_chunk_size: usize,
    ) -> Self {
        if total_len >= min_total_len && chunk_size >= min_chunk_size {
            Self::Parallel
        } else {
            Self::Sequential
        }
    }
}

/// Iterates over mutable `chunk_size`-element chunks of `data`, passing
/// `(index, chunk)` to `f` through the selected dispatch provider.
#[inline]
pub fn for_each_chunk_mut_enumerated<T, F>(
    data: &mut [T],
    chunk_size: usize,
    dispatch: ChunkDispatch,
    f: F,
) where
    T: Send + Sync,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    match dispatch {
        ChunkDispatch::Parallel => {
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Parallel, _, _>(
                data, chunk_size, f,
            );
        }
        ChunkDispatch::Sequential => {
            data.chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(i, chunk)| f(i, chunk));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{for_each_chunk_mut_enumerated, ChunkDispatch};

    #[test]
    fn workload_threshold_selects_dispatch_mode() {
        assert_eq!(
            ChunkDispatch::for_workload(4096, 512, 4096, 512),
            ChunkDispatch::Parallel
        );
        assert_eq!(
            ChunkDispatch::for_workload(4095, 512, 4096, 512),
            ChunkDispatch::Sequential
        );
        assert_eq!(
            ChunkDispatch::for_workload(4096, 511, 4096, 512),
            ChunkDispatch::Sequential
        );
    }

    #[test]
    fn sequential_and_parallel_dispatch_preserve_chunk_index_semantics() {
        let mut sequential = vec![0usize; 16];
        let mut parallel = vec![0usize; 16];

        for_each_chunk_mut_enumerated(&mut sequential, 4, ChunkDispatch::Sequential, |i, chunk| {
            chunk.fill(i);
        });
        for_each_chunk_mut_enumerated(&mut parallel, 4, ChunkDispatch::Parallel, |i, chunk| {
            chunk.fill(i);
        });

        assert_eq!(sequential, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
        assert_eq!(parallel, sequential);
    }
}
