# ADR 0050: Cache-line traversal for fused twiddle multiplication

- **Status:** Rejected
- **Date:** 2026-09-06
- **Class:** [patch]
- **Item:** [APOLLO-FOUR-STEP-CACHE-TILE](../../backlog.md#apollo-four-step-cache-tile)

## Context

The generic FourStep operation uses 16-by-16 scalar tiles for its fused
multiply-transpose. The retained phase instrument places layout work at
80–92% of N=262144 latency. The register-tile experiment in
[ADR 0049](0049-fused-twiddle-transpose.md) establishes no accepted complete-
transform gain and grows the executable; that candidate is removed.

The existing scalar operation reads source and twiddle rows contiguously,
then writes each product at a matrix-row stride. Changing the traversal order
can alter the set of simultaneously live destination cache lines without
changing the operation count or buffer footprint. Cache conflicts are a
hypothesis: neither the phase profiler nor elapsed time measures cache misses.
For f64 complex rows of length 256/512, destination addresses are 4096/8192
bytes apart; a smaller column tile reduces simultaneously revisited destination
lines. Whether those strides collide in cache sets depends on actual cache
geometry and placement, so stride arithmetic alone is not conflict evidence.

## Provider contract and prerequisite

Locked Leto `a2006adec7f522f27a359160b8f692bd85e22972` exports
`leto_ops::cached_cache_geometry() -> CacheGeometry` and
`CacheGeometry::cache_line_bytes(self) -> usize`. The former uses a process-
wide `LazyLock`, rather than a topology query per transform. Apollo already
depends on Leto ops with its default `topology` feature.

Leto aggregates Themis `CacheLevel` values: minimum reported capacity per
level and maximum reported line width across levels and cache instances.
This is process-wide geometry, not a selected-core cache size. If topology
or line widths are absent, its documented line-width fallback is 64 bytes.
Leto's analogous `line_elements_for<T>` is private to generic array traversal;
there is no public FFT-specific tile policy to reuse.

The metadata-only release diagnostic reports 37 cache records on the Intel
Core Ultra 9 285K. Performance CPU 1 has 48 KiB L1 data and private 3 MiB L2;
efficiency CPU 3 has 32 KiB L1 data and 4 MiB L2 shared by processors 2–5.
Both share a 36 MiB L3 with all 24 logical processors. Every reported level
covering the selected processors has a 64-byte line. Leto's process-wide
policy reports 32 KiB L1, 3 MiB L2, 36 MiB L3 and a 64-byte line, consistent
with its minimum-capacity/maximum-line policy.

Evidence is `output/apollo-cache-tile/cache-topology.txt` with source/lock
identity in `topology-manifest.json`. All-target Clippy passes; the ignored
release diagnostic passes in 3.944 seconds under the unchanged Nextest bound.
These are topology observations only: phase timings from this run are not a
performance baseline. WMI aggregate block sizes do not supply these facts.

## Evaluated candidate

The candidate selects the scalar tile side once before the fused traversal:
`(cached_cache_geometry().cache_line_bytes() / size_of::<Complex<F>>()).clamp(1, 16)`.
The lower bound makes iterator steps nonzero; the upper bound preserves the
existing maximum tile dimensions. The scalar contract admits nonzero-sized
complex values. The reported 64-byte line gives four f64 complex values or
eight f32 complex values per side on this host.

Keep the current loop body, multiplication order, twiddle table, output
indices and `.min` tile boundaries. Each output element remains
`out[c * rows + r] = source[r * columns + c] * twiddle[r * columns + c]`.
Products are independent, so reordering tiles introduces no reduction-order
change. Existing edge clipping covers any non-power-of-two reported line
width. Do not add scalar-specific bodies, new buffers, register staging,
ISA dispatch or tile-width specialization families.

The closure is private: `four_step/execution.rs::decompose` selects the fused
loop; `four_step_fft` serves forward/inverse and recursive odd-power callers,
including multidimensional lane execution. Selection, row submissions, first
and final transpose and workspace calculation remain unchanged. Applicable
behavioral tests are `four_step/tests/workspace.rs` (independent impulse
spectrum and normalization), `plan/fft/lanes/tests.rs` (static/dynamic,
strided and recursive lanes), and `kernel/retained_footprint/submissions.rs`
(caller-owned scratch lifetime). The phase diagnostic remains unchanged
apart from recording the prerequisite topology outside measured regions.

## Measurements and stop criterion

The baseline binds provider-integration commit `1de31e26` and lock SHA256
`0CE26A23EB9A909220515BF76539FFB6930C05B7AD0B33CA5BB40E728F8DC6E1`.
The old ADR 0049 binaries do not establish the new dependency baseline.

Run the unchanged engine census with its committed 60-second supervisor on
retained matched binaries. Use two replications of both execution orders for
each queried core class and the existing family-wise interval comparator.
Record process-level concurrent load with floating-point CPU deltas; an
exclusive lease is coordination, not evidence of actual isolation. Invalidate
contaminated sets, including unexplained compiler activity, and never count
phase-only gains as complete-engine speedup.

Acceptance requires a supported complete-transform improvement, no supported
regression, unchanged analytical results and allocation bounds, and no
executable-size increase. Clippy and release Nextest retain their committed
budgets. Inspect codegen for the extra cache-policy read, bounds and loop
control, without inferring zero overhead from generic syntax. The workload,
input sizes, tolerances and benchmark body remain fixed.

This is one tile-policy experiment, not a parameter sweep. Reject it if a
correctness/allocation gate fails, complete-transform regression is supported,
artifact size grows without an attributed bounded correction, or the valid
replicated measurements establish no benefit. Preserve the diagnostic and
rejected evidence; remove an unproven production candidate. A later change
to fusion, routing, provider APIs or scratch ownership requires a separate
scope and decision.
## Result and decision

Reject the candidate and retain the original 16-by-16 scalar traversal.
All-target Clippy and 545 release library tests pass. Sixteen unchanged census
runs finish within the 60-second bound, but the valid performance-core
replicated comparison establishes neither supported improvement nor regression
across 39 cases. Between-run spread leaves one RustFFT composed case and one
unchanged Apollo 1024 case undecidable. No supported effect justifies a further
size-correction experiment.

The efficiency-core set is invalid for performance conclusions: the last
run's endpoint interval contains active `cargo-clippy` PID 25004 (0.046875 CPU
seconds), plus two idle Cargo processes. Process ownership is not established.
The timing lease therefore does not establish actual isolation. No replacement
run is needed to decide this candidate. The earlier baseline-phase output is
also excluded because hook activity had not been explicitly released; the
candidate phase measurement provides attribution only, without a paired phase
speedup claim. Endpoint process snapshots miss short-lived processes.

The executable grows from 6,862,336 to 6,865,408 bytes (+3,072). PE sections
attribute 2,688 bytes to code, 160 to read-only data, 24 to unwind records and
4 to relocations; file alignment accounts for the remainder. Inspected f64
assembly calls the cached geometry getter at fused-loop entry and emits
runtime division for variable-step loop trip counts. These observations do
not assign the entire size delta to individual symbols or establish zero
abstraction overhead.

Cold memory differs from warm memory. The unchanged retained-footprint probe
warms the pool, then records table/plan/first-forward/warm-forward windows over
the same length ladder. At the first generic N=65536 forward, the candidate
performs 83 global allocations versus 6 after restoration. Peak window bytes
are 2,108,904 versus 2,101,692: 77 extra allocations and 7,212 transient bytes.
Retained bytes match at 2,101,692. Other ladder cases match, every warm window
allocates zero bytes, and direct Mnemosyne allocation counts remain zero.
This is workload evidence for first-use topology initialization, with arbitrary
worker placement still uncontrolled; cached lookup syntax alone never implies
allocation-free first use. The census independently shows cold peak growth,
while its warm allocation results remain unchanged.

After restoration, the independent impulse and static/dynamic, strided and
recursive lane tests pass (four tests, 0.213 seconds). The rebuilt census
returns to 6,862,336 bytes; every byte outside COFF/debug timestamps and
CodeView PDBAge matches the retained baseline. No production change remains.
A standard Nextest archive preserves the restored phase-test binary for later
experiments without source or build-cache swapping.

Evidence is under `output/apollo-cache-tile`: `census-manifest.json` records
source/lock identity, runner, comparison mapping and validity; `binary-sizes.json`
and `binary-sections.txt` attribute size; `candidate-footprint.txt` and
`restored-footprint.txt` preserve cold/warm windows; `candidate-execution.rs`
identifies the rejected source; `restored-production-identity.json` and
`restored-tests.tar.zst` preserve restoration evidence. Raw CSV rows retain all
39 cases and 100 ordered samples; the benchmark workload is unchanged.

## Revision

2026-09-06: changed Proposed to Rejected after replicated measurements,
independent contention review, cold-allocation attribution and restored
production verification. Cache-line arithmetic alone does not establish
profitable traversal or eliminate the first-use topology cost.
