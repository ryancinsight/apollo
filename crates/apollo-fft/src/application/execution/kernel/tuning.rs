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
/// Derived by `pot::crossover`, which runs both routes at one length in one
/// process with the cache flushed before each arm and the arm order
/// alternating. That instrument exists because the previous figure was measured
/// one route per process, where the between-process difference exceeded the
/// between-route difference it was resolving; see ADR 0039's revision note.
///
/// Set from measurements that control for the hybrid scheduler, because the
/// previous value was set by measurements that did not. On the Core Ultra 9
/// 285K, Windows hands benchmark child processes EcoQoS — efficiency cores at
/// efficiency frequency, the thread wandering between them — and every
/// "process-dependent" four-step anomaly in this audit was that, not code.
/// `pot::core_matrix` pins the thread and removes the scheduler from the
/// question entirely; at N = 4096 four-step beats Stockham on **both** core
/// types (16.6 us against 28.1 on an E-core, 13.2 against 62.6 on a P-core),
/// and `pot::crossover` — both routes, one process, cache flushed per arm —
/// has it ahead from N = 256 through 2^20.
///
/// 256 rather than 4096 since the f64 sized codelet arms also consult this
/// predicate: `codelet::pinned_probe` measured the batched four-step against
/// the sized route itself at 256 and 1024 — 1.5x faster on an E-core, 5 to 6x
/// on a P-core — completing the pinned evidence down to the crossover the
/// in-process instrument reported. See ADR 0039's fifth revision note.
pub(crate) const ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD: usize = 1 << 8;

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
