# ADR 0040: Leto FFT layout ownership

- Status: Accepted
- Date: 2026-08-26
- Class: [patch] [arch] in Apollo; breaking provider migration in Leto
- Item: [APOLLO-FOUR-STEP-SQUARE-MOVEMENT](../../backlog.md#apollo-four-step-square-movement)
- Earlier drivers: `ATLAS-APOLLO-LETO-LAYOUT-PASSES-2026-08-26`,
  `ATLAS-APOLLO-LETO-VIEW-LAYOUT-2026-08-27`,
  `ATLAS-APOLLO-HERMES-COMPLEX-TRANSPOSE-2026-09-01`

Revision 2026-09-08: retain the accepted CPU layout boundary and extend it to
FourStep's pure-copy transposes. The `3f1c0db7` consumer graph passes its local
retention criteria. Leto PR 175 lands as `d9ca3252`; Apollo's joint provider
adoption at `54e8f2d8`, lock `0ECC20AB`, also passes the unchanged acceptance.

## Context and ownership

Static and dynamic multidimensional FFT plans duplicated tiled gather/scatter
loops for non-contiguous axes. FourStep also owned scalar and AVX transpose
implementations. These operations move values without FFT arithmetic, while
Leto owns shape, stride, assignment and layout movement in Atlas.

Apollo owns decomposition, lane scheduling, scratch lifetime, twiddles, sign
and normalization. Leto owns value-preserving movement; Hermes supplies its
register operations and hardware dispatch. Hephaestus remains the GPU provider.
This decision changes no Apollo public API, scalar support or transform
convention and introduces no GPU algorithm change.

## Decision and contracts

Public multidimensional entry points execute C-dense views directly, including
offset C-dense views. Fortran-dense and general strided views stage once in
logical C order, execute the complete transform, then assign logical indices
back through Leto. Physical memory order is not the transform's logical order.

Staging borrows a rank-disjoint existing plan-scratch role: rank two uses the
otherwise dormant 3-D X role; rank three uses the otherwise dormant 2-D role.
That role is unreachable from the rank's nested axis passes. No additional
full-volume scratch slot or temporary transposed allocation is introduced.

Internal axis passes use `ComplexLayout::transpose_complex_matrices` with
adjacent row-major `[rows, columns]` sources and `[columns, rows]` destinations.
The 2-D plan uses one matrix; the 3-D Y pass uses adjacent `[ny, nz]` planes;
the X pass treats the volume as one `[nx, ny * nz]` matrix. Reverse passes
exchange the dimensions to restore row-major ordering.

FourStep uses the same batch operation with count one at both out-of-place
sites and `ComplexLayout::transpose_square_inplace` at the square in-place
site. Its caller-owned workspace is clipped to the required extent before
decomposition. The private `MixedRadixScalar` bound requires `ComplexLayout`,
which supplies `LaneScalar + Pod`; the supported implementor set is unchanged.
The former scalar transpose hook, its two overrides and the scalar/AVX copy
modules are deleted without forwarding functions. Fused twiddle/transposition
arithmetic remains Apollo-owned and unchanged.

For source `(r, c)`, `r * columns + c` equals the Fortran-view offset of `(c, r)`
in shape `[columns, rows]`. Copying those logical indices to row-major storage
therefore transposes the matrix; exchanging dimensions twice is an involution.
Provider preflight checks dimension products and exact source/destination
lengths before mutation. Empty batches and zero-area matrices do no assignment.
Square tile pairs are loaded before either store; diagonal tiles transpose in
place, and each ragged border pair moves once. Movement performs no scalar
arithmetic and preserves complete complex representations.

Leto's four concrete scalar implementations call one private generic movement
family. A non-generic square entry owns each scalar's instantiation; the batch
entry exposes count/extent specialization. Apollo's concrete cold failure
functions preserve invariant messages, typed error Debug output and tracked
caller locations without duplicating generic error-formatting paths. These
are internal impossible-state failures, not alternate valid-input execution.
Leto's removed free functions and migration are specified by
[provider ADR 0027](../../../leto/docs/adr/0027-hermes-complex-batch-transpose.md).

## Alternatives and causal evidence

- Keeping Apollo's copy family duplicates provider indexing, tile selection,
  tails and tuning. Allocating Leto arrays instead violates warmed allocation
  bounds. General axis iteration discards the known dense transpose structure;
  Leto's checked canonical dense-copy operation preserves it.
- Executing Fortran-dense views in physical order is incorrect for rectangular
  logical lanes. The original 2-D/3-D comparisons produced errors of 3.10/7.38;
  staged-versus-C-order exact-value tests reject this semantic error without
  relaxing floating-point tolerances.
- Source deletion alone does not guarantee smaller code. Count-one calls do
  not select the high-count register path, and generic assignment retained
  layout validation and error cleanup. The checked dense-copy boundary removes
  that machinery without removing preflight. Exposing validation removed
  divisions but did not alone satisfy executable size acceptance.
- A generic/default scalar-role method or generic `inline(never)` does not
  remove multiple consumer instantiation roots. Retained linker maps observed
  duplicate square bodies; concrete provider entries reduced them to one per
  ISA. Library count-one specialization and a general census call already
  coexisted before that change. See the
  [linked attribution](../../../../output/apollo-square-transpose/preflight-inline/map/attribution.md).
- A shared checked tile-span formulation added runtime division and AVX-512
  payload staging; it was rejected by the
  [codegen comparison](../../../../output/apollo-square-transpose/tile-span/codegen.md).
  Consolidating provider tile-range diagnostics reduced some duplication but
  produced a supported complete-engine regression. Other gains did not excuse
  it; the [rejected comparison](../../../../output/apollo-square-transpose/tile-diagnostics/audit-summary.json)
  remains the falsifying evidence, not a causal attribution of that regression.
- Apollo's concrete error boundary instead removed duplicate Debug/drop roots
  while preserving square and batch bodies. The
  [linked comparison](../../../../output/apollo-square-transpose/cold-failures/map/normalized-comparison.json)
  and [caller-location decoding](../../../../output/apollo-square-transpose/cold-failures/map/caller-locations.json)
  support that mechanism. Its executable fell to 6,861,312 bytes, 512 below the
  6,861,824-byte baseline; static identity alone supplies no latency result.

## Retention and verification

Retain the production change only when unchanged behavioral and allocation
oracles pass, the linked movement code introduces no new payload spills or
divisions, the executable does not grow, and the complete-engine comparison
supports an improvement without any supported regression. A size-rejected
candidate's timing informs diagnosis only. Change direction when those
falsifiers hold; neither favorable isolated cases nor source-level elegance
overrides them. Do not change the workload, sampling or thresholds to pass.

The retained comparison uses 16 prescribed baseline/candidate invocations,
39 cases and 100 ordered samples per case across two processor classes,
counterbalanced orders and replications. Complete-family rank-33/68 intervals
must separate across every comparison and the full between-run enclosure.
Paired median percentage ranges are descriptive, not confidence intervals.
The suite remains within 300 seconds and each invocation within 60 seconds.
The statistical assumptions and source-only baseline preparation contract
belong to [ADR 0036](0036-native-benchmark-regression-oracle.md): each revision
retains its production manifests/lock; identical instrument source does not
imply identical transitive provider locks. Control drift invalidates timing.

Behavioral oracles cover rectangular/ragged/multi-plane permutations, offsets,
whole-buffer canaries and NaN/infinity/signed-zero/subnormal payload bits for
both Apollo precisions. Provider coverage additionally includes F16/Bf16 and
the private scalar path. Exact and oversized FourStep workspaces use the same
impulse-spectrum error bound and preserve unused suffixes; short storage and
overflow reject before mutation. Static/dynamic 2-D/3-D direct-DFT, normalized
round-trip, Fortran/strided parity and C-dense pointer-identity tests guard
composition and staging. The warmed allocation census rejects temporary
allocation. Standalone locked consumer gates and provider audits guard drift.

Public API comparison must use each revision's exact lock. The checker's
independent-manifest generation previously selected the wrong Leto revision
and failed before comparison. The retained
[JSON comparison](../../../../output/apollo-square-transpose/integration/api/compare-cache-access-result.json)
passes 223 applicable checks, with 31 skipped, for archived tree `304d70d8`
against its recorded baseline. Both JSON files use nightly-2026-08-01 format
61 and cargo-semver-checks 0.50.0; 25 manifests were compared separately.
JSON mode does not compare manifests. This result is historical, not a fresh
SemVer verdict for the current graph.

## Evidence boundaries

### Accepted historical consumer graph

Revision `3f1c0db7` uses Leto `633acb7` and lock
`D43E38E8A3C55A976E6FDA7B77685086C60B24987278339EA3ACFB9C20023D42`.
The [collected gates](../../../../output/apollo-square-transpose/integration/provider-graph/collection.json)
include 1,458 all-feature workspace tests excluding Python (38 skipped),
564 release FFT library tests (37 skipped), Clippy, seven doctests (one
ignored), warning-denied docs, seven bounded smokes and supply-chain checks.
Default-feature docs also exclude Python. Cargo-deny retains 32 configured
duplicate warnings; this is not warning-free dependency evidence.

The [independent census audit](../../../../output/apollo-square-transpose/integration/provider-graph/census/independent-audit.json)
accepts all 16 runs, 624 rows and 62,400 samples with unchanged artifact
identities, a 6,861,312-byte executable and an 83.508-second suite. It supports
one gain and no regression: efficiency-core full-real length 262,144 has
baseline median envelopes 2.2790–2.9478 ms versus 2.0837–2.2058 ms. Four paired
median reductions span 7.32–26.16%, not a confidence interval. Direct competitor
comparisons have four leads, fourteen losses and two overlaps; no general
superiority over RustFFT or PhastFT is established.

All warmed allocation/retained records and the separate 20 default-feature
f64 footprint windows match. Efficiency-core cold length-65,536 maxima differ
by 24 bytes; identical cold behavior is not established. The footprint probe
warms the process pool. Its live-byte windows exclude caller signal buffers,
may retain caches across sizes, and measure neither process-cold memory nor
OS resident memory. Other precisions/features need their own footprint evidence.

[PE inspection](../../../../output/apollo-square-transpose/integration/provider-graph/census/pe-code-identity.json)
matches 6,630 retained movement/review instruction rows; 691 other code bytes
differ in prior-map debug-formatter regions. Whole-image identity is not
claimed. Geometry captured during unmeasured prewarm is test-only; virtual
addresses do not establish physical cache mapping. Phase shares are attribution,
not speedup or cache-miss measurements. Endpoint load guards passed within a
coordinated compiler-free interval, but miss inaccessible CPU totals and
transient processes. Caller affinity does not pin Moirai workers.

### Current provider adoption

Leto PR 175 lands as `d9ca3252`; joint adoption with Hephaestus `f6f55f45`
resolves the old default-runtime requirement conflict. Lock
`0ECC20ABAF30420CD543A13141557BE266AC5AB1AD94A0A39CF5C3B602BF66C8`
selects default Moirai 0.6 `5c8a9e8`, default Mnemosyne `82d3daa1` and
Aequitas `a442d16`. Eunomia, Hermes and Themis identities remain unchanged.
Moirai adds eight Mnemosyne packages pinned to `2eb49c1`; the old direct
runtime retains its eight-package `7f173751` memory pin. There are three
Mnemosyne identities and two Moirai identities; total packages rise 307→314.

The `2eb49c1` pin belongs to Moirai's WASM provider co-evolution. Its manifest
gives no concrete removal condition; correcting that provider declaration and
stale ADR is upstream work, not a downstream source patch. Distinct runtime
and allocator sources carry distinct nominal types and state. The public Leto
partition API's Moirai 0.6 types cannot accept Apollo's direct 0.5 types.

Apollo retains direct Moirai `83aa411` because
[source inspection](../../../../output/apollo-square-transpose/integration/provider-graph/moirai-closure.json)
finds its worker-idle hook absent from default `5c8a9e8`. Allocator maintenance
cannot release Apollo's live worker-thread scratch. Removing the pin requires
an upstream owner-thread idle capability and the existing reclamation oracles;
it must not delete reclamation or substitute an adapter. Keeping the direct
hook does not install it in the other runtime pool.

The [collected adoption gates](../../../../output/apollo-square-transpose/integration/provider-adoption/collection.json)
bind 19 passing commands to HEAD `4fbdeb26` plus this lock. They cover format,
safety, all-target/all-feature Clippy, 1,458 workspace tests with 38 skips and
564 release FFT tests with 37 skips. No test crosses the 30-second slow bound.
Seven doctests pass with one ignored; public workspace documentation denies
warnings. Both exclude Python. Seven benchmark smokes, provider and advisory
checks pass; cargo-deny retains 32 configured duplicate warnings. All 20
default-feature allocation windows exactly match baseline counts and bytes.
The [locked API comparison](../../../../output/apollo-square-transpose/integration/provider-adoption/api/manifest.json)
passes 223 checks with 31 skipped against `9da1f9f7`, using the two original
locks and JSON format 61. All 25 manifests match after line-ending normalization.
The
[retained build](../../../../output/apollo-square-transpose/integration/provider-adoption/census/collection.json)
is 6,861,312 bytes, but its build record is not a timing result.
[Current PE inspection](../../../../output/apollo-square-transpose/integration/provider-adoption/census/pe-code-identity.json)
finds nine of eleven retained movement ranges identical at the same addresses.
The other two differ in four address operands: two relocated callees preserve
their normalized instruction rows and 55 referenced read-only bytes match.
The whole code section has 215,012 differing aligned bytes; remaining program
and runtime-state differences are unclassified. All 1,102 tracked compiler
inputs remain fixed through gate collection;
only this ADR's prose changes.

The [independent current-graph audit](../../../../output/apollo-square-transpose/integration/provider-adoption/census/independent-audit.json)
accepts all 16 runs and 62,400 samples in 83.445 seconds within the reserved
compiler-free interval and unchanged 300-second suite bound. One supported
gain remains after charging the complete-family and between-run spread:
efficiency-core real half-spectrum length 262,144 has median envelopes
1.4389–1.9973 ms before versus 1.2802–1.3558 ms after. Four paired median
reductions span 9.88–23.52%; this is a descriptive range, not a confidence
interval. No supported regression or unchanged-control difference is found.

Warm and retained records match, as do the separate 20 footprint windows.
Cold length-65,536 maxima increase by 58 bytes on the performance core and
48 bytes on the efficiency core; cold-process equivalence is not established.
No compiler endpoint includes rustdoc or other compiler/linker processes.
The observed 6.93–11.11% host CPU utilization excludes 247–249 surviving
processes with unreadable totals and cannot bound entirely transient work.
Caller affinity still does not pin runtime workers.

[Current direct competitor envelopes](../../../../output/apollo-square-transpose/integration/provider-adoption/census/competitor-envelope.csv)
contain four efficiency-core leads: PhastFT at 1,024 and 4,096, and both
PhastFT and RustFFT at 16,384. Thirteen comparisons lose and three overlap;
this is no general superiority claim. The audit preserves raw output,
including the inherited `DiagnosticOnly` runner label, and owns the acceptance
verdict for this size-passing graph.

The retained-executable Atlas runner extension used by this experiment is
verified local work whose upstream commit remains pending. Its executable
hash is retained with the run; this evidence does not establish availability
of that option from the published Atlas tooling.

No Python extension runtime, Miri/sanitizer, physical AVX-512 or unexecuted
platform coverage is implied. Output links follow Atlas's 14-day/10-GiB
retention policy; they are experiment evidence, not release artifacts.
