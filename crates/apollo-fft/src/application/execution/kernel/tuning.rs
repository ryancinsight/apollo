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
/// **The isolated crossover is not the operating crossover, and the difference
/// is the whole reason this constant is 65536.** `pot::crossover` puts
/// four-step ahead of Stockham from `N = 256` upward, by 2 to 3x. Setting the
/// threshold there reproduces ADR 0039's rejected measurement exactly: 4096
/// degrades from 29 to 348 us and 16384 from 275 us to 1.64 ms in
/// `benches/engine_census`.
///
/// Both are real. Timing the kernel from inside the census process shows it
/// genuinely taking 99 us per call at `N = 4096` where the same binary's test
/// process takes 12, so the route is not mismeasured — it is slower there.
/// Four-step touches three `N`-sized arrays at once (data, scratch, and the
/// `W_N^(j*k)` matrix) against Stockham's two, and appears sensitive to how
/// those land relative to each other, which differs by allocation history.
///
/// So the retained value is the one measured in a process that resembles a
/// caller. Raising it on the isolated figure would ship a 12x regression at
/// 4096. See ADR 0039's 2026-08-26 revision and
/// `ATLAS-APOLLO-FOUR-STEP-LAYOUT-SENSITIVITY-2026-08-26`.
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
