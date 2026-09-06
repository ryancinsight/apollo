# ADR 0039: One-dimensional power-of-two routing

- **Status:** Accepted
- **Date:** 2026-08-26
- **Class:** [patch] [arch]
- **Item:** [APOLLO-FOUR-STEP-TWIDDLE-RETENTION](../../backlog.md#apollo-four-step-twiddle-retention)
- **Revision:** 2026-09-05, reconcile the current routes and their table ownership.

## Context

The one-dimensional planner selects small codelets, register/base transforms,
Stockham, or four-step decomposition. The scalar and layout determine which
kernel implements a route. Atlas owns the substrate: Hermes supplies CPU lanes,
Leto supplies host views and transpose operations, Moirai supplies parallel row
execution, and Mnemosyne supplies reusable scratch storage.

Earlier versions of this record mixed successive crossover decisions with
contradicted explanations for timing variation. The current code is the source
of the route description below. Historical experiments remain in Git; their
absolute timings are not current performance guarantees.

At the September 5 entry baseline, generic four-step plans eagerly acquired a
full-length forward stage table. Their inverse executors acquired the matching
inverse table. `FourStep::run` ignored both arguments: the decomposition acquired
its own row, matrix, or odd-power combine tables. For a square split this retained
an unused `(N - 1)`-element complex table per direction in the process cache.

## Decision

1. `one_dimensional_uses_four_step` remains the shared route predicate. Its
   current threshold is 256, and four-step admits both even and odd powers of
   two. This change does not retune that threshold.
2. Existing direct, sized and supported base routes through 1024 retain their
   selection and required tables. In particular, sized f32 execution retains
   its Stockham behavior; the generic plan change does not reroute it.
3. Generic static and dynamic plans select four-step before acquiring stage
   tables. The dynamic plan records that route explicitly. Its executor invokes
   the existing four-step operation with the selected direction and normalization.
4. Four-step owns every table it consumes. Even powers use square decomposition
   with row/matrix or planar tables. Odd powers use two even-power halves and a
   radix-2 combine; that combine still acquires its required full-length table.
5. Normalized inverse execution applies the existing full-length normalization.
   No arithmetic, transform sign, scalar precision, row scheduling or scratch
   policy changes.

The public API and provider roles remain unchanged. Mnemosyne's merged
scratch-release APIs allow removal of the expired PR 128 revision quarantine;
Cargo.lock remains the standalone source pin. The resource saving
is the unused table payload, `(N - 1) * size_of::<Complex<T>>()`, per executed
direction for even generic powers. At N=65536 that is 1,048,560 bytes for f64 or
524,280 bytes for f32, excluding allocation and map overhead. Odd powers have
no claimed table-payload saving.

## Alternatives

- **Keep eager tables and ignore them:** retains process-lifetime memory and
  performs cold construction for data the selected operation never reads.
- **Delete tables by exponent parity in the planner:** duplicates decomposition
  knowledge. Table acquisition belongs to the operation that consumes it;
  the odd-power combine already provides that ownership boundary.
- **Change sized or base routes at the same time:** would combine resource
  ownership with a separate performance decision and enlarge the regression
  surface. Their existing selection remains intact.
- **Replace the provider stack:** does not address unused Apollo plan data.

## Verification contract

Sparse complex inputs at indices zero and N/4 have an analytical spectrum whose
phase cycles through the fourth roots of unity. Generic tests apply that oracle
to both native scalar widths, static and dynamic plans, and forward, normalized
inverse and unnormalized inverse execution. Lengths cover the base boundary,
even and odd powers, and the parallel-row boundary. Their floating-point bounds
derive from unit roundoff, radix depth and input magnitude.

The reference census records cold peak, retained bytes and warmed allocation
for Apollo, RustFFT and PhastFT using unchanged signal sizes. RustFFT keeps its
planner-sized scratch with the plan, matching reusable execution; its convenience
`process` method would otherwise allocate scratch on each measured call.

Allocation measurements establish memory behavior, not throughput. A speed
claim additionally requires matched executable evidence and stable reference
controls. The N=32 unchanged-executable experiments exhibit between-run drift
larger than their within-run confidence intervals; neither a planner-state nor
an EcoQoS explanation is established by those observations.

The September 5 Windows x64 census uses Rust 1.97.0 and the Atlas development
overlay. Four fresh runs per executable reproduce these retained-byte counts
for forward complex f64 execution, excluding caller signal storage:

| N | Before | After | RustFFT with scratch | PhastFT |
|---:|---:|---:|---:|---:|
| 1024 | 37,228 | 37,228 | 32,832 | 15,552 |
| 4096 | 206,368 | 140,832 | 131,072 | 64,896 |
| 16384 | 731,216 | 469,072 | 524,384 | 261,504 |
| 65536 | 4,140,288 | 3,091,612 | 2,097,536 | 1,048,320 |
| 262144 | 11,542,544 | 7,348,240 | 8,388,928 | 4,194,048 |

Each engine's warm 1-D peak increment is zero. Both census executables occupy
6,779,904 bytes. These are allocation-accounting results for this workload,
not process RSS bounds. The replicated 39-case timing comparison finds no
supported regression; two unchanged small cases remain inconclusive because
between-run spread exceeds the effect. No speedup is established.

## Revision history

- **2026-08-26–27:** establish shared routing, then revise the one-dimensional
  crossover to 256 using the route instruments. Subsequent code adds odd-power
  decomposition and supported base routes.
- **2026-09-01:** correct inverted core-class labels and withdraw the unmeasured
  EcoQoS causal claim. Those corrections remain binding on historical evidence.
- **2026-09-05:** replace contradictory accumulated route descriptions with the
  current selection and table-ownership contract; remove eager unused generic
  stage tables under the linked backlog item. Exact commands and measurements
  belong to that item's delivery evidence.
