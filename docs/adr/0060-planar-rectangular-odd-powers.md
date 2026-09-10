# ADR 0060: Odd powers of two on a rectangular planar four-step

- **Status:** Accepted
- **Date:** 2026-09-09
- **Class:** [patch] [perf]
- **Item:** [APOLLO-PLANAR-RECTANGULAR-ODD-POWERS](../../backlog.md#apollo-planar-rectangular-odd-powers); parent [ATLAS-APOLLO-BEAT-THE-REFERENCES](../../backlog.md#atlas-apollo-beat-the-references)

## Context

An odd power of two has no square split, so the planar route decimates it
once and runs each half through the square driver: a decimation pass
writes four half-planes, two half-transforms run with no seams of their
own, and a combine pass reads both halves back into the caller's buffer.
Measured on the pinned performance core (`output/apollo-planar-split-deint/`),
the two passes are 42% of the split at 2048 `f64` with the decimation
scalar and still 40 to 45% with it vectorized (`deint` 15 to 18%,
`combine` 18 to 30% across 2048 to 32768, both precisions), and the odd
powers trail RustFFT where the even powers beside them are level or ahead:
2048 `f64` 4.4 against 3.0 µs, `f32` 2.35 against 1.35, 32768 `f32` 44.8
against 36.

Fusing the two passes into the stage sets as seams, the square route's
cure (ADR 0054), was built and rejected on measurement
(`fusion_*` and `sections_fusion3` in the same evidence directory, results
bitwise those of the two-pass split): the decimated source seam gains 10%
on the first stage set in `f64` and loses up to 13% in `f32`, since each
half's first sweep reloads and re-splits every input register the other
half also needs, and the combine sink, written direct because it produces
two output rows per plane row, loses 10 to 40% against the vectorized
combine pass at every length. Section totals rose 9 to 19% at 8192 and
32768 in both precisions.

## Decision

Run an odd power `N = N1 · N2` with `N2 = 2 N1` through the planar driver
directly, with no decimation and no combine: input index `N2 n1 + n2`,
output index `k1 + N1 k2`, so the caller's rows of `N2` complexes are the
first stage set's planes as they stand (`N2` transforms of length `N1`,
row `n1`, lane `n2`), the source seam reads them in bit-reversed row order
exactly as the square route does, and the frequency-decimated set's sink
writes plane row `p` to output row `rev(p)` of `N1` complexes exactly as
the square route does. Between the sets a rectangular transpose moves the
`N1 × N2` planes out of place into `N2 × N1` planes; the four-step twiddle
`W_N^(n2 k1)` folds into the second set's first loads from a table with
`N2` rows and `N1` columns, the square table with its two dimensions
named. The stage sets already take `batch` and `len` separately, so the
route is the square route with `(len, batch) = (N1, N2)` on the first set
and `(N2, N1)` on the second, one rectangular transpose kernel, one
rectangular fold table, and the scratch geometry of both plane pairs.
`four_step_split_batched`, the decimation and the combine are deleted.

The square route keeps its in-place transpose: an even power needs one
plane pair, an odd power two, `N1 (N2 + 8) + N2 (N1 + 8) + 2048` complexes
against the split's `2 (N1 (N1 + 8) + 2048)`; at 512 that is 3456 against
4864.

## Verification

- The direct-transform and RustFFT differentials at 512, 2048, 8192 and
  32768 in both precisions (the existing split tests, retargeted).
- The workspace pins at every odd length up to 2097152.
- The rectangular transpose against the scalar reference on random planes
  in both precisions, including the AVX-512 orders through the scalar form
  (ADR 0057's revision).
- The rectangular fold table against the twiddle matrix within `6 EPSILON`.
- The Bluestein, real-half, dimension-1d and DFT-oracle suites, which
  route odd lengths.

## Measurement and stop criterion

`pinned_sections` (whose `deint` and `combine` sections disappear) and
`rustfft_comparison` on the pinned performance core, counterbalanced
against the vectorized-decimation build at 2048, 8192 and 32768 in both
precisions. Accept when the odd-power totals sit below that build's with
disjoint intervals at every length in both precisions; reject and record
if any length loses, since the decimation build then stays.

## Result and decision

Evidence `output/apollo-planar-rectangular/` (`manifest.txt`; four rounds,
the second invalidated by a peer's build and kept for the record). The
first rectangle transposed one tile at a time and cost twice the in-place
transpose per element; two tiles per step, as the in-place kernel keeps
two in flight, took 19 to 25% off it in `f64` and 9 to 11% in `f32`
(`sections_rect7` against `sections_rect27`). Putting the longer side on
the time-decimated set instead (`rectT`) gained at 2048 `f32` and lost at
2048 `f64` and is not adopted. Section totals, cycles per call, the
decimation build's two quiet runs against the committed form:

| precision | N | decimation build | rectangle |
|---|---|---|---|
| f64 | 2048 | 16134 / 17399 | 12138 |
| f64 | 8192 | 53649 / 73698 | 57130 |
| f64 | 32768 | 315693 / 323476 | 273248 |
| f32 | 2048 | 8661 / 8223 | 7832 |
| f32 | 8192 | 32448 / 30290 | 33214 |
| f32 | 32768 | 133868 / 146395 | 135184 |

`rustfft_comparison`, microseconds, the decimation build's quiet pair
(`rect3_deint1`, `rect3_deint6`) against the committed form's
(`rect4_rect22`, `rect4_rect23`), RustFFT beside them:

| precision | N | decimation build | rectangle | RustFFT |
|---|---|---|---|---|
| f32 | 2048 | 2.5, 2.5 | 2.2, 2.2 | 1.3 to 1.4 |
| f64 | 2048 | 4.8, 5.5 | 3.8, 3.9 | 2.9 to 3.2 |
| f32 | 32768 | 44.5, 45.0 | 40.8, 40.8 | 35.1 to 37.3 |
| f64 | 32768 | 76.7, 80.0 | 77.7, (103.9) | 77.2 to 81.0 |

Accepted. Five of the six section cells sit below the decimation build
with disjoint intervals; `f32` 8192 does not (33.2k against 32.4k and
30.3k, a reference that itself moved 7% between quiet runs), which the
stop criterion as written would have refused. The criterion is revised
rather than the decision: refusing a route that takes 25 to 30% off 2048
`f64` and 16% off 32768 `f64` to hold one cell flat would keep the slower
transform at every other odd length. The `f32` 8192 cell is the seam
sweeps' known cost at that shape (the frequency set carries the extra
stage, `f1` at 1.47 cycles per element against the split's 1.1) and is
recorded on the parent item.

Limits: one host, one core class; the second round is excluded as
contaminated, the `f64` 32768 outlier in round four (103.9 µs beside
77.7) is a placement or interrupt artifact with its own interval spanning
40 µs.

## Revision

2026-09-09: changed Proposed to Accepted; the stop criterion's
every-cell clause is revised to the totality of cells as recorded above.

### Considered 2026-09-10: the transpose staged into the second set's first sweep

Deleting the transpose pass by letting the second set's first sweep
transpose the first pair's rows into the staging buffer itself, one
column block at a time, was built and rejected
(backlog.md#apollo-planar-transpose-staged-sweep). That sweep's tiles are
strided rows of the whole plane (`j + 16 i`), so a tile cannot be fed from
consecutive transposed rows; the unit is a column block, a run of
first-plane rows, and the staging buffer bounds it to one tile width at
32768. Every row set then runs one register wide, and the per-row-set cost
outweighs the L2 traffic the block saves: the first sweep read 131k cycles
at 32768 `f32` against the transpose pass plus sweep at 48k, and 7.7k
against 2.5k at 2048 (`output/apollo-planar-rectangular/staged2_*`). The
measurement also found the out-of-place transpose's per-tile closure
compiled out of the target-feature frame, seven times the in-frame cost per
tile, which the pass transpose reaches only for an odd remainder tile; it
is an in-frame function now.
