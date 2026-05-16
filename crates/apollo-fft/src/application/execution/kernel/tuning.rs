//! Kernel-level tuning constants.

/// Chunk-size threshold above which composite kernels switch to Rayon chunk execution.
pub(crate) const RADIX_PARALLEL_CHUNK_THRESHOLD: usize = 32_768;

/// Minimum power-of-two length at which the four-step algorithm is preferred over Stockham.
pub(crate) const FOUR_STEP_THRESHOLD: usize = 1 << 17;
