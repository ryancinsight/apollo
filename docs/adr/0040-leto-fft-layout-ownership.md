# ADR 0040: Leto FFT layout ownership

- **Status:** Accepted
- **Date:** 2026-08-26
- **Class:** [patch] [arch]
- **Items:** `ATLAS-APOLLO-LETO-LAYOUT-PASSES-2026-08-26`,
  `ATLAS-APOLLO-LETO-VIEW-LAYOUT-2026-08-27`

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
2. Each internal non-contiguous FFT axis pass is presented as a Leto rank-two
   assignment. A row-major source with shape `[rows, columns]` is viewed as a
   Fortran-contiguous matrix with shape `[columns, rows]`, then assigned to a
   row-major destination with that same transposed shape. The destination is
   caller-owned scratch, so no intermediate allocation occurs.

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
and 3-D errors of `3.10` and `7.38` against direct-transform bounds below
`5e-12`. This is a semantic mismatch, not a floating-point tolerance issue.

## Correctness and performance contract

For row-major source element `(r, c)`, the linear offset is
`r * columns + c`. A Fortran-contiguous view of shape `[columns, rows]` maps
logical element `(c, r)` to `c + r * columns`, the same offset. Assigning that
view to a row-major destination therefore produces the mathematical transpose.
Repeating the operation with exchanged dimensions restores the original
ordering.

The transpose helper accepts only exactly sized source and destination slices
and checks dimension multiplication before constructing views. Empty batches
and zero-area matrices perform no assignment. The entry helper preserves
logical indices for any valid injective mutable layout. The selected staging
role is unreachable from that rank's nested axis passes, so it remains live
without adding another full-volume scratch slot. All scratch remains
thread-local and reused by the plan scratch bank.

The controlled provider benchmark compares Leto assignment with Apollo's
superseded loops in one binary at identical addresses. Four of eight
gather/scatter confidence intervals are disjoint in Leto's favour and four
overlap; none favours the old loop. Apollo's repeated end-to-end census is more
variable across uncontrolled host runs, so it establishes value semantics,
zero warm allocations, and a diagnostic baseline, not an isolated layout
speedup.

## Failure modes and verification

- Swapped dimensions or matrix counts fail rectangular, ragged-tile, and
  multi-plane transpose tests.
- Tail loss fails the 35x67 and 67x35 cases, which cross the provider's tile
  edge in both orientations.
- Incorrect axis composition fails static and dynamic two- and
  three-dimensional direct-DFT and round-trip tests.
- Confusing physical and logical order fails Fortran-dense rectangular cases;
  rejecting non-dense input fails strided cases; copying C-dense input fails
  the offset-view pointer-identity case.
- A temporary allocation fails the warmed allocation census.
- Provider drift fails the standalone locked build and provider audit.
