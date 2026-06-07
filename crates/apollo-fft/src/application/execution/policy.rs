//! Execution policy abstractions for zero-cost static dispatch over parallelism regimes.

use rayon::prelude::*;

use std::future::Future;

/// Zero-cost execution policy trait mapping parallelism strategies to compile-time
/// monomorphized variants.
pub trait ExecutionPolicy: Send + Sync + 'static {
    /// Async abstraction
    type Future<T: Send>: Future<Output = T> + Send;

    /// Iterates over mutable chunks of `data`, passing `(index, chunk)` to `f`.
    fn for_each_chunk_mut_enumerated<T, F>(data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send + Sync,
        F: Fn(usize, &mut [T]) + Send + Sync;
}

/// Sequential (synchronous) execution policy.
pub struct SyncPolicy;

impl ExecutionPolicy for SyncPolicy {
    type Future<T: Send> = std::future::Ready<T>;

    #[inline]
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

/// Data-parallel execution policy using Rayon.
pub struct ParallelPolicy;

impl ExecutionPolicy for ParallelPolicy {
    type Future<T: Send> = std::future::Ready<T>;

    #[inline]
    fn for_each_chunk_mut_enumerated<T, F>(data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send + Sync,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        data.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    }
}

use std::pin::Pin;
use std::task::{Context, Poll};

/// Wrapper around Tokio's JoinHandle that unwraps the result to fulfill `Future<Output = T>`.
pub struct AsyncFuture<T>(pub tokio::task::JoinHandle<T>);

impl<T: Send> Future for AsyncFuture<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.0).poll(cx) {
            Poll::Ready(Ok(val)) => Poll::Ready(val),
            Poll::Ready(Err(e)) => std::panic::resume_unwind(e.into_panic()),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Asynchronous execution policy using native Tokio tasks.
pub struct AsyncPolicy;

impl ExecutionPolicy for AsyncPolicy {
    type Future<T: Send> = AsyncFuture<T>;

    #[inline]
    fn for_each_chunk_mut_enumerated<T, F>(data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send + Sync,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        // Tokio does not provide a native data-parallel array iterator.
        // For synchronous mutable chunk processing within an async context,
        // we execute sequentially since this method cannot `.await`.
        data.chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    }
}
