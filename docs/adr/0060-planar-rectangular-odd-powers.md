# ADR 0060: Odd powers of two on a rectangular planar four-step

- **Status:** Proposed
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
