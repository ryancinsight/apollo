//! Kernel-level tuning constants.

/// Chunk-size threshold above which composite kernels switch to Moirai chunk execution.
pub(crate) const RADIX_PARALLEL_CHUNK_THRESHOLD: usize = 32_768;

/// Minimum power-of-two length at which the four-step algorithm is preferred over Stockham.
///
/// At N=4096 (k=12) the sub-DFTs are N1=N2=64 elements (1 KiB per f64 sub-problem),
/// trivially L1-resident. The Stockham path for N≥4096 requires 3–4 passes over a
/// working set that exceeds typical L1, so the four-step's cache-resident
/// sub-problems win for lane-transform callers of the general dispatcher.
/// Standalone 1-D plans use the separately measured
/// [`ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD`].
pub(crate) const FOUR_STEP_THRESHOLD: usize = 1 << 12;

/// Measured crossover for standalone one-dimensional power-of-two plans.
///
/// This currently equals the four-step row-parallel threshold because the
/// route loses until its independent rows amortize Moirai orchestration. It is
/// a distinct decision so scheduler tuning cannot silently reroute transforms.
pub(crate) const ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD: usize = 1 << 16;

/// Maximum value of `prev_len * R_TOTAL` for which a fused Compose stage fires.
///
/// Bounds the intermediate buffer allocated from the thread-local `COMPOSE_ARENA`
/// in `Compose::compute_group`.  The arena pre-grows to 2 × FUSE_THRESHOLD ×
/// sizeof(Complex<f64>) ≈ 2 MB on the outermost call and is reused thereafter.
///
/// Active chains at FUSE_THRESHOLD = 65536:
///
/// | Chain | R_TOTAL | max prev_len | reduction target |
/// |-------|---------|-------------|-----------------|
/// | C16   | 65536   | 1           | 2^16 → 1 pass   |
/// | C15   | 32768   | 2           | 2^15×m → 1 pass |
/// | C14   | 16384   | 4           | 2^14×m → 1 pass |
/// | C13   | 8192    | 8           | 2^13×m → 1 pass |
/// | C12   | 4096    | 16          | 2^12×m → 1 pass |
/// | C11   | 2048    | 32          | …               |
/// | C10   | 1024    | 64          | …               |
pub(crate) const FUSE_THRESHOLD: usize = 65536;
