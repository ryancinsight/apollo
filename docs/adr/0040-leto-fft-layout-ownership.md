# ADR 0040: Leto FFT layout ownership

- **Status:** Accepted
- **Date:** 2026-08-26
- **Class:** [patch] [arch]
- **Items:** `ATLAS-APOLLO-LETO-LAYOUT-PASSES-2026-08-26`,
  `ATLAS-APOLLO-LETO-VIEW-LAYOUT-2026-08-27`,
  `ATLAS-APOLLO-HERMES-COMPLEX-TRANSPOSE-2026-09-01`

**Revision 2026-09-06:** [APOLLO-FOUR-STEP-SQUARE-MOVEMENT](../../backlog.md#apollo-four-step-square-movement)
evaluates extending the accepted layout boundary below to every pure-copy
FourStep transpose. **No candidate in this experiment is accepted or merged.**
Retention requires supported complete-engine improvement without supported
regression, unchanged allocation bounds and no executable growth. Diagnostic
timing of a size-rejected candidate informs the next bounded hypothesis; it
does not relax acceptance.

### Experimental boundary and verification

Leto owns in-place complex square movement and the existing
`transpose_complex_matrices` operation. Apollo's experimental FourStep calls
use the latter with matrix count one at both out-of-place sites and the square
operation at the in-place site. `MixedRadixScalar<Complex = eunomia::Complex<F>>`
already implies `LaneScalar + Pod`. Exact factor-sized slices and caller
workspace clipping require no conversion or additional scratch role. The
private scalar transpose hook, two scalar overrides and scalar/AVX copy
modules are removed without a forwarding layer. FFT arithmetic, fused twiddle
multiplication, decomposition, routes, sign and normalization remain unchanged;
Apollo's public API and implementor set do not change.

The moved provider-contract tests preserve the original Cartesian product of
shapes, offsets, special-value payloads and whole-buffer canaries for both
precisions. Typed errors replace the removed private kernel's panic contract;
short storage and overflow must leave destination bytes unchanged. Zero-area
cases borrow empty windows. Every original impulse case now covers exact and
oversized caller workspaces with the same analytical bound and an untouched
suffix. Provider tests cover f32, f64, F16 and Bf16, including the scalar path.

[Candidate 6 consumer gates](../../../../output/apollo-square-transpose/pure-copy/apollo-gates/final-checks.json)
pass 1,442 debug tests, 552 release FFT tests, Clippy, documentation, seven
bounded benchmark smokes and all 20 matched default-feature allocation windows.
[The locked tile-span provider gates](../../../../output/apollo-square-transpose/tile-span/leto-gates/final-checks.json)
pass 923 native tests, nine focused cases in debug and release, Clippy, minimal
features, 27 doctests (one existing ignored), rustdoc and 24 smoke cases.
The earlier additive square API passes 196 SemVer checks against `a2006ad`;
subsequent private traversal revisions do not change its public contract.
Hermes `07c5e5f` passes 548 native and 16 release tests. These gates establish
behavior and specified memory windows, not performance acceptance; unavailable
ISA execution, Python extension runtime tests and Miri/sanitizer coverage are
not claimed by the consumer record.

### Candidate comparison

The retained [baseline](../../../../output/apollo-square-transpose/baseline.json)
is 6,861,824 bytes. Each size below comes from the unchanged census executable
build scenario; P and E denote its selected performance and efficiency logical
processors. Percentage ranges describe four paired sample medians. Supported
directions additionally require rank-33/68 intervals to separate across both
execution orders and both replications for the complete 39-case family.

| Candidate | Bounded change | Executable bytes; delta | Result and retained evidence |
|---|---|---:|---|
| 1 | Initial square register tiles | 6,876,160; +14,336 | Size rejected; no timing. [Build](../../../../output/apollo-square-transpose/array-construction/candidate.json) |
| 2 | Seed one register row, fill the exact remainder; lazy overflow error | 6,874,624; +12,800 | Size rejected; no timing. [Build](../../../../output/apollo-square-transpose/seeded-array/candidate.json) |
| 3 | Hermes facet inlining and exact fill slice | 6,873,600; +11,776 | Size rejected; no timing. Outlined helpers and complete-tile payload spills eliminated. [Codegen](../../../../output/apollo-square-transpose/feature-frame/codegen.md) |
| 4, Leto `ce9d02b` | Shared scalar tail outside ISA frames; narrow extent error | 6,868,992; +7,168 | Size and performance rejected: P real-full/1024 slower by 0.95–1.51%; no supported gain. [Census](../../../../output/apollo-square-transpose/census-manifest.json), [regression](../../../../output/apollo-square-transpose/performance/regressions-stderr.txt), [memory/load audit](../../../../output/apollo-square-transpose/audit-summary.json) |
| 5, Leto `6013768` | Restore 16-by-16 outer cache blocks around register tiles | 6,871,552; +9,728 | Size rejected. E real-half/262144 improves by 17.40–35.35%; no supported regression. [Audit](../../../../output/apollo-square-transpose/cache-blocking/audit-summary.json) |
| 6 | Route all FourStep pure copies through existing Leto APIs | 6,892,544; +30,720 | Size and performance rejected: P complex/4096 slower by 2.64–11.92%, P real-full/16384 slower by 11.60–26.58%; E real-half/262144 improves by 15.14–41.35%. [Independent audit](../../../../output/apollo-square-transpose/pure-copy/audit-summary.json) |
| 7, Leto `f3a6dd8` | Checked shared tile spans and disjoint row slices | 6,902,272; +40,448 | Size and codegen rejected: +9,728 versus candidate 6; AVX-512 adds runtime division and payload spills. Provider behavioral gates pass; consumer build/assembly only, no timing. [Build](../../../../output/apollo-square-transpose/tile-span/candidate.json), [codegen](../../../../output/apollo-square-transpose/tile-span/codegen.md) |
| 8, Leto `3ad43b7` | Checked canonical `transpose_copy` boundary and lazy layout errors | 6,872,576; +10,752 | Size rejected. E complex/65536 improves by 12.65–20.62%; no supported regression or P-core direction. Warm allocations and retained bytes match. [Independent audit](../../../../output/apollo-square-transpose/dense-copy/audit-summary.json) |
| 9, Leto `9a47d6b` | Inline existing extent validation across crates | 6,872,576; +10,752 | Size rejected. E complex/65536 improves by 16.55–17.81%, E real-half/262144 by 29.90–38.30%; no supported regression or P-core direction. [Independent audit](../../../../output/apollo-square-transpose/preflight-inline/audit-summary.json) |
| 10, Leto `437b502` | Shared private tile bounds-failure diagnostic | 6,870,016; +8,192 | Size and performance rejected: E complex/1024 slower by 0.59–4.22%; E complex/65536 improves by 14.36–17.83% and real-half/262144 by 27.17–33.63%. No supported P-core direction. [Audit](../../../../output/apollo-square-transpose/tile-diagnostics/audit-summary.json) |

Candidates 4–6 each retain 16 runs, 39 unique eight-field cases and 100 ordered
samples per case. Native output and extracted CSV match; executable,
instrument, runner, comparator and lock hashes match their manifests. Their
suites take 90.704, 90.906 and 91.040 seconds respectively within 300 seconds;
every invocation passes its existing 60-second bound. Candidate 4's narrow
regression clears the complete enclosure: candidate lower 1,570,350 ps exceeds
baseline upper 1,567,021 ps. The audits preserve the other full envelopes and
paired medians; they establish retained-binary differences, not their cause.

Across these censuses all warmed allocation signatures and retained-byte
ranges match. Cold 65,536-point peaks have overlapping, nonidentical ranges;
no exact cold equivalence is claimed. The census measures global allocator
live-byte windows, not OS resident memory: cold includes plan construction and
first execution at each ordered size, retained is the ending live balance,
and warm is one further call. Caller signal buffers are excluded and caches
can persist across sizes. The separate footprint probe warms the process pool
before its ladder and therefore is not cold-process evidence.

Endpoint process snapshots show no compiler/linker, but inaccessible CPU
totals remain null and new/dead or entirely transient processes have no complete
interval CPU bound. The audits use double CPU differences, DateTimeOffset
elapsed intervals and 24 logical processors; low observed load is not proof
of an uncontended interval. Caller affinity does not pin Moirai workers.

### Code-generation evidence and remaining hypothesis

Historical phase attribution places the final transpose near 18% of a generic
262,144-point execution; candidate 4's retained diagnostic places it near 21%.
Neither is a matched speed comparison or a cache-miss measurement. The retained
[baseline assembly](../../../../output/apollo-transpose-isa/library.s)
(`54D2023E...05832E4DFA`) emits one `Complex<f64>` pair-swap specialization:
two bounds branches per pair, 16-byte matrix loads/stores and a 16-byte stack
store/reload. Its paired addresses advance by 16 and `16 * side` bytes
(4,096/8,192-byte strides at sides 256/512). It does not establish a standalone
f32 specialization or measured cache traffic. Square register exchange instead
loads both off-diagonal tiles before writing, transposes diagonal tiles in
place and handles each ragged border element once.

Construction and code-placement changes remove eager success-path extent
errors, outlined array/permutation helpers and duplicated scalar tails; the
narrow square error also avoids rejection-time allocation and unrelated solver
diagnostic dependencies. These mechanisms have codegen and behavioral evidence,
but their candidate executables still grow.

[Candidate 6 assembly](../../../../output/apollo-square-transpose/pure-copy/codegen.md)
shows why source deletion alone is insufficient. Matrix count one excludes
Leto's high-count register route; generic assignment retains borrowed-layout
and validation machinery. Its body has 1,476 instructions plus 354 cleanup
instructions, versus 370 in the canonical mover. The removed Apollo dispatcher
and AVX family total 656 instructions; the new provider/generic/mover bodies
total 2,044 before cleanup and supporting layout/error functions. These are
library instruction counts, not exact linked-byte attribution or valid-input
allocation counts. Eager overflow-error drops also remain on success paths.
Candidate 8 targets that checked dense boundary through the existing mover,
preserving batch validation, dispatch, error behavior and all value tests;
its provider verification passes 930 native tests, 366 release tests, 28
doctests (one existing ignored), Clippy, documentation, the 24-case smoke and
196 SemVer checks per package. Its retained assembly removes the generic
assignment and cleanup family and all ordinary successful-path LetoError drop
calls. The new 16-run census takes 90.967 seconds within 300 seconds and
supports the E complex/65536 gain in the table: candidate rank-68 upper
476,500,000 ps falls below baseline rank-33 lower 518,100,000 ps. Endpoint
snapshots do not bound inaccessible or transient load. Size acceptance still
fails, so the prepared full Apollo gate remains pending.

Candidate 9 exposes existing length/extent validation
to cross-crate compilation. Candidate 8 retained two pairs of
alternative-width chunk-count divisions in the count-one copy path. Checks,
error values and ordering stay intact. Division/loop removal must avoid cold
code duplication and text growth; this hypothesis does not promise recovery
of the remaining 10,752-byte gap. Candidate 9 removes both targeted batch
chunk-count division pairs and both validator calls. Five inherited row-division
pairs remain. The copy body falls from 535 to 437 instructions; its frame grows
from 296 to 328 bytes. Text falls by 32 bytes, while the executable remains
6,872,576 bytes. These code-generation observations do not establish an FFT
speedup. The 16-run complete-engine census takes 90.105 seconds and supports
the two E-core directions in the table. Candidate upper bounds are 448,400,000
ps for complex/65536 and 1,394,300,000 ps for real-half/262144, below baseline
lower bounds of 514,500,000 and 1,635,400,000 ps respectively. Warm allocation
signatures and retained-byte ranges match; cold peaks overlap but differ.
[Whole-library attribution](../../../../output/apollo-square-transpose/preflight-inline/codegen.md)
finds a net 619 emitted instructions above baseline, including register kernels,
copy validation, formatting and unwind cleanup. It is not linked-byte ownership;
the completed [linker-map diagnostic](../../../../output/apollo-square-transpose/preflight-inline/map/identities.json)
matches all 4,801,601 code bytes, their section address and image base against
the timed candidate. Only PE headers and read-only data differ. The map exposes
distinct Apollo-library and census-crate instantiations of the same f64 square
operation. Their instantiating-crate symbol suffixes differ; this establishes
duplicate instantiation roots, not ThinLTO import as their cause. Linked body
and relocation comparisons show identical AVX2/AVX-512 instruction streams
after normalizing address displacements, with identical external call targets.
Separate identical shuffle constants and `tile.rs:30:47` panic-location records
distinguish the copies. Their redundant pair occupies a 4,032-byte interval
including padding; this is not a guaranteed saving or an attribution of all
growth. [Linked attribution](../../../../output/apollo-square-transpose/preflight-inline/map/attribution.md)
records the exact evidence and limits.

Candidate 10 consolidates the private load/store tile-range failure into one
non-generic cold diagnostic. Checked slice access preserves the safe boundary;
the failure carries the offending range and storage length. Public typed errors,
tile construction, movement, ISA coverage and workloads remain unchanged. The
falsifier is retained per-instantiation location data, new spills/divisions,
text growth or a supported complete-engine regression. Removing the full
10,752-byte gap is not assumed. A generic `inline(never)` attribute is rejected
as an ownership fix because the two instantiating-crate roots would remain. Provider
revision `437b502` passes Clippy, format and all 30 unchanged focused tests in
debug and release. Independent source review finds no defect. The consumer
[linked map](../../../../output/apollo-square-transpose/tile-diagnostics/codegen.md)
confirms one folded AVX2 square body and one diagnostic helper; AVX-512 copies
retain distinct constant references. No new divisions or payload spills appear.
The unchanged 16-run census takes 90.141 seconds and triggers the performance
stop condition: E complex/1024 candidate lower 2,977,157 ps exceeds baseline
upper 2,975,000 ps. Other supported gains do not override this failure. Warm
allocation signatures and retained bytes match. The candidate is rejected;
forward restoration `00665a4` matches candidate 9 tile access exactly and passes
the unchanged 30 provider tests in debug and release, Clippy and format.
Candidate 10 evidence remains intact. This comparison establishes a regression,
not its cause. Full consumer restoration gates pass; the prior size failure
remains a merge blocker.

The restored consumer executable is 6,872,576 bytes, still 10,752 above the
baseline. Its 4,801,601-byte `.text` payload has the same SHA256 as retained
candidate 9; the complete executable hash differs. The
[identity record](../../../../output/apollo-square-transpose/tile-diagnostics/restoration/text-identities.json)
establishes code-byte equality, not a fresh timing result. No timing rerun
replaces candidate 10's rejection evidence.

The [restoration gates](../../../../output/apollo-square-transpose/tile-diagnostics/restoration/apollo-gates/final-checks.json)
pass 1,442 native tests, 552 release tests, six focused tests, seven doctests
(one existing ignored), Clippy, rustdoc, format, the safety ratchet and seven
bounded smoke cases. All 20 default-feature retained-memory windows match
the accepted baseline. Thirty-five compiled input hashes and the lock remain
fixed throughout; subsequent board/ADR changes only record results. Configured
Python exclusions and unavailable sanitizer/ISA coverage remain explicit.

The unchanged bounded phase probe also completes in 3.598 seconds. Its three
blocks place fused multiply/transpose at 38.0–38.7% of P-core and 36.7–37.4%
of E-core time for complex f64 length 262,144; pure-copy movement remains
material. This is phase attribution, not a baseline speedup comparison or
cache-miss measurement. Worker placement, inaccessible processes and transient
load remain uncontrolled, as recorded with the
[probe](../../../../output/apollo-square-transpose/tile-diagnostics/restoration/phase-profile/result.json).

The existing probe now observes actual borrowed data/scratch addresses, byte
extents and row strides during its first unmeasured prewarm call. Timing and
geometry capture are mutually exclusive and share the same RAII reset; the
hook and capture types exist only under `cfg(test)`. The offset-slice oracle
and unchanged two-size/two-scalar/two-core probe pass in release (two tests,
eight geometry records), with format and Clippy passing. The
[observation record](../../../../output/apollo-square-transpose/tile-diagnostics/restoration/buffer-observation/rationale.md)
states the source/lock identity and transient peer overlay reconciliation.
Virtual addresses establish borrowed-buffer geometry, not physical cache
mapping or a speedup. No timing workload or production algorithm changes.

[Direct competitor envelopes](../../../../output/apollo-square-transpose/preflight-inline/competitor-envelope.csv)
reuse the same complete-family rank intervals across four candidate runs per
core. Apollo trails RustFFT at all five P-core sizes and all E-core sizes except
16,384. It leads PhastFT at E-core sizes 1,024, 4,096 and 16,384. At 262,144 it
trails both competitors on both cores. P-core PhastFT comparisons at 4,096,
16,384 and 65,536 overlap. These results describe this retained candidate and
machine, not a stack-wide performance claim.

The current experimental lock selects Leto `00665a4` through two entries and
Hermes `07c5e5f` through five, without changing manifest requirements or
registry selections. [Leto](../../../leto/backlog.md#leto-square-transpose) and
[Hermes](../../../hermes/backlog.md#hermes-complex-permutation-inlining) remain
review-branch dependencies: provider merges precede accepted consumer delivery;
rejection removes the unaccepted candidate and temporary lock selections.
The restored candidate 9 implementation is in this lock. Retained output links follow Atlas's
14-day/10-GiB policy and contain experiment manifests, not release artifacts.

**Revision 2026-09-01:** Leto Ops PR #135, merged as `060eb7eb`, added one
public allocation-free batched-complex transpose. It selects the widest exact
Hermes hardware width among 16/8/4 scalar lanes for the measured high-count
small-matrix regime and retains Leto's generic assignment for every other
shape or target. Apollo now delegates its one private CPU axis-transpose
boundary to that provider instead of reconstructing a Leto view pair per
matrix. Apollo retains plan-owned scratch and Moirai axis scheduling; it owns
no register-tile implementation or capability probe.

**Revision 2026-08-27:** The first implementation established Leto ownership
for internal transpose passes but admitted public mutable views through
`as_mut_slice_memory_order`. That accessor returns physical order for both C-
and Fortran-dense layouts, while Apollo's axis kernels require logical C order.
The corrected boundary executes C-dense views directly and stages every other
layout once through a rank-disjoint reusable scratch role before assigning the
result back through Leto.

## Context

Apollo's two- and three-dimensional FFT plans apply one-dimensional kernels
along non-contiguous axes. The plans gathered each axis into reusable scratch,
executed contiguous lane transforms, and scattered the result back. Static and
dynamic plans each contained their own tiled index loops for those layout
passes even though the loops performed no FFT arithmetic.

Leto owns array shape, stride, and assignment semantics in the Atlas stack.
Provider PR 125, merged as `1e70b27e`, made rank-two assignment use one
canonical kernel and added a tiled C-destination/Fortran-source transpose.
Retaining Apollo's copies after that provider change would duplicate both the
layout policy and its performance tuning.

## Decision

Apollo owns FFT decomposition, lane scheduling, scratch lifetime, twiddle
selection, sign, and normalization. Leto owns value-preserving layout movement.
Two private Apollo helpers enforce that boundary:

1. The public multidimensional view entry exposes a C-dense block directly,
   including offset C-dense views. Fortran-dense and general strided layouts
   are assigned into a C-order view backed by a rank-disjoint plan-scratch
   role. Rank two borrows the otherwise dormant 3-D X role; rank three borrows
   the otherwise dormant 2-D role. The complete transform runs there before
   Leto assigns logical indices back to the caller's layout.
2. Each internal non-contiguous FFT axis pass calls Leto Ops' batched complex
   transpose with adjacent row-major `[rows, columns]` sources and row-major
   `[columns, rows]` destinations. Leto performs complete preflight, selects
   the Hermes register-tile or canonical assignment route, and writes directly
   into Apollo's caller-owned scratch without intermediate allocation.

The two-dimensional plan uses one matrix. The three-dimensional Y pass uses a
batch of adjacent `[ny, nz]` planes. Its X pass treats the volume as one
`[nx, ny * nz]` matrix. Reverse assignments exchange the dimensions and restore
the original row-major layout.

This changes no public API, transform convention, scalar support, or GPU
execution. Hephaestus remains the GPU provider; this decision covers the CPU
layout boundary used by Apollo plans.

## Rejected alternatives

### Keep Apollo's tiled copies

Rejected because the four copies in static and dynamic execution encode the
same transpose that Leto now owns. They duplicate indexing, tile selection,
tail handling, and future tuning work.

### Use Leto's general axis iterators

Rejected for this contiguous rank-two case. The provider's assignment kernel
recognizes the exact C-destination/Fortran-source layout pair and executes its
tiled transpose directly. General strided iteration would discard that
structural information.

### Allocate transposed arrays

Rejected because Apollo already owns reusable scratch sized for the complete
plan. A temporary Leto allocation would violate the established zero-allocation
warm execution contract.

### Execute Fortran-dense views in physical order

Rejected because physical-order chunks do not represent row-major logical
lanes for a rectangular Fortran layout. The old implementation produced 2-D
and 3-D errors of `3.10` and `7.38` relative to C-order execution of the same
plans. The corrected layout tests require bit-for-bit staged-versus-C-order
parity, so this is a semantic mismatch rather than a floating-point tolerance
issue. Separate C-order tests retain direct-DFT and normalized round-trip
coverage for the transform algorithm.

## Correctness and performance contract

For row-major source element `(r, c)`, the linear offset is
`r * columns + c`. A Fortran-contiguous view of shape `[columns, rows]` maps
logical element `(c, r)` to `c + r * columns`, the same offset. Assigning that
view to a row-major destination therefore produces the mathematical transpose.
Repeating the operation with exchanged dimensions restores the original
ordering.

The provider accepts only exactly sized source and destination slices and
checks dimension multiplication before selecting a kernel. Empty batches and
zero-area matrices perform no assignment. The public-view entry helper
preserves logical indices for any valid injective mutable layout. The selected
staging role is unreachable from that rank's nested axis passes, so it remains
live without adding another full-volume scratch slot. All scratch remains
thread-local and reused by the plan scratch bank.

The controlled provider benchmark compares Leto's generic assignment with its
Hermes-backed batched operation in one binary at identical addresses. Both
runs improve every measured f32/f64 small-matrix case. Apollo's unchanged
100-sample engine census reduces the selected f64 4,096x4x4 3-D median from
the 1.1567 ms entry to 263.225/265.350 us (77.24%/77.06%) while retaining zero
warmed allocations for every measured 2-D and 3-D shape. These timings are
local Windows AVX2 evidence; AArch64 is compile-only evidence.

## Failure modes and verification

- Swapped dimensions or matrix counts fail rectangular, ragged-tile, and
  multi-plane transpose tests.
- Tail loss fails the 35x67 and 67x35 generic cases and the 256x15x13
  register-path batch; the 256x16x16 case covers complete provider tiles.
- Incorrect axis composition fails static and dynamic two- and
  three-dimensional direct-DFT and round-trip tests.
- Confusing physical and logical order fails Fortran-dense rectangular cases;
  rejecting non-dense input fails strided cases; copying C-dense input fails
  the offset-view pointer-identity case.
- A temporary allocation fails the warmed allocation census.
- Provider drift fails the standalone locked build and provider audit.
