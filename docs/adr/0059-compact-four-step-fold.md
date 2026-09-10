# ADR 0059: The four-step fold from a two-level table

- **Status:** Accepted
- **Date:** 2026-09-09
- **Class:** [patch] [perf]
- **Item:** [APOLLO-FOUR-STEP-COMPACT-FOLD](../../backlog.md#apollo-four-step-compact-fold); parent [ATLAS-APOLLO-BEAT-THE-REFERENCES](../../backlog.md#atlas-apollo-beat-the-references)

## Context

The fold sweep multiplies every element by `W_N^(p k)` on the first pass
of the frequency-decimated set, and read that matrix from two planes as
large as the data itself: 1 MiB at 65536 `f64`, 4 MiB at 262144, streamed
from L2 or beyond on every call. ADR 0056's attribution put the fold sweep
at twice the plain sweep at 65536 and at 767k to 1034k cycles against the
plain sweep's 318k at 262144, where the planes, the fold and the caller's
buffer total 12 MiB against a 3 MiB L2.

## Decision

Plane column `k` holds memory column `c = order(k)`, and `c = G F + f` with
`F` the lane group and `f = order(k mod F)`, so `W_N^(p c) = W_N^(p F G) ·
W_N^(p f)`. `FourStepFold` holds a coarse table of one scalar per row and
lane group and a fine table of one register per row: `m (F + m / F)`
entries in place of `m^2`, 128 KiB against 1 MiB at 65536 `f64`, 640 KiB
against 4 MiB at 262144. The fold pass loads one fine register per row
per step and broadcasts the group's coarse twiddle, one complex multiply
more per element. Every entry is one direct evaluation through the shared
twiddle authority, so an entry is within `4u` of the exact twiddle (each
factor within one ulp, two roundings in the product) against `u` for the
full matrix, which the transform's `O(log N · u)` forward-error bound
absorbs; the full-matrix builder is deleted with its only consumer, and
the interleaved oracle's own cache keeps its direct construction.

The second multiply pays for itself only where the full matrix would
stream from beyond L2, so the two-level form applies from
`COMPACT_FOLD_MIN_LEN = 2^16` (2^18 until the 2026-09-10 revision below); below it the fine table is the full row and
the coarse table one, and the pass takes the single multiply on a
loop-invariant test.

## Verification

The batched suite, the four-step workspace suite, the RustFFT differential
at 65536 and 262144 (which bounds the fold's added rounding), the
dimension-1d plan suite, the crate's API tests and the DFT oracle sweep.
Two table tests hold every entry of the product against the interleaved
matrix within `6 EPSILON` (three one-ulp evaluations and the product's two
roundings; worst measured `4.03`), at 256 (full row) and at 262144
(two-level).

## Measurement

`pinned_sections` on the pinned performance core, the staged-sink build
against the two-level fold at every length (`fold_stage1` / `fold_fold1`),
thousands of cycles per call:

| precision | N | fold sweep before | after | total before | after |
|---|---|---|---|---|---|
| f64 | 16384 | 25 | 27 | 119 | 121 |
| f64 | 65536 | 123 | 117 | 555 | 525 |
| f64 | 262144 | 635 | 369 | 4174 | 3905 |
| f32 | 16384 | 12 | 14 | 60 | 62 |
| f32 | 65536 | 48 | 58 | 247 | 257 |
| f32 | 262144 | 217 | 156 | 1812 | 1766 |

The two-level form halves the fold sweep at 262144 in both precisions and
loses 8 to 20% of it at 16384 and 65536 `f32`, where the full matrix sits
in L2 and the second multiply is the whole cost; hence the bound. With the
bound, and the table's form a compile-time parameter of the pass with
affine row offsets (a run-time test and an offset table in the row loop
each cost the full-row form 15% at 65536 `f64`), the lengths below it read
as before and 262144 keeps the gain (`fold_stage6` / `fold_fold6`, second
pair): `f64` 65536 fold sweep 120k against 124k, total 548k against 549k;
`f64` 262144 fold sweep 713k against 363k, total 4398k against 4134k;
`f32` 262144 fold sweep 222k against 142k, total 1892k against 1785k;
`f32` 65536 total 256k against 262k, within the run-to-run spread.

## Revision 2026-09-10: the bound moves to 2^16

Driven by [APOLLO-FOLD-COMPACT-MEMORY](../../backlog.md#apollo-fold-compact-memory):
the retained-footprint census read the full-row tables as `16n` bytes per
direction below the bound, so the trade was re-measured with memory on the
table. Section probe (`planar_passes_by_size`, performance core, four runs
base/compact/compact/base, thousands of cycles, the better of each pair),
the retained blocks after the first forward (`retained_footprint_attribution`)
and the peak working set (`engine_census`), evidence in
`output/apollo-fold-compact-2026-09-10/`:

| scalar | n | fold sweep full | fold sweep compact | total full | total compact | tables full | tables compact | peak full | peak compact |
|---|---|---|---|---|---|---|---|---|---|
| f64 | 16384 | 24.7 | 28.9 | 117.3 | 122.7 | 256 KiB | 72 KiB | 471 KB | 281 KB |
| f64 | 65536 | 142.1 | 117.1 | 571.8 | 547.8 | 1 MiB | 272 KiB | 1860 KB | 1086 KB |
| f32 | 16384 | 12.9 | 14.3 | 61.4 | 61.2 | | | | |
| f32 | 65536 | 53.1 | 58.6 | 259.0 | 260.1 | | | | |

At 65536 the compact form is faster (`f64`, 0.96 of the transform) or level
(`f32`, 1.00) and the tables shrink 3.8 times (the `f64` lane width is four,
so `m (F + m / F)` against `m^2` is `4 + 64` against `256` per row); the
bound moves to 2^16. At 16384 the compact form costs 5% of the `f64`
transform for 184 KiB per direction and the bound keeps the full row. The
original measurement above (2026-09-09) read the 65536 `f64` transform
level and the `f32` sweep 20% behind; this pass re-measured the trade with
the table's memory on it and reads the 65536 sweep faster at `f64` and
the transform level at `f32`.

## Consequences

262144 `f64` gains about 6% of its whole transform and `f32` about 3%. The
remaining fold cost at 65536 is the pass itself, not its table; the source
sweep and the fold sweep's traffic are the prefetch item's.
