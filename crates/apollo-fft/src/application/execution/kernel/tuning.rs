//! Kernel-level tuning constants.

/// Chunk-size threshold above which composite kernels switch to Rayon chunk execution.
pub(crate) const RADIX_PARALLEL_CHUNK_THRESHOLD: usize = 32_768;
