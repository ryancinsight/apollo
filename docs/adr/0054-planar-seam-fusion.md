# ADR 0054: Planar seams fused into the stage passes

- **Status:** Accepted
- **Date:** 2026-09-08
- **Class:** [patch] [perf]
- **Item:** [APOLLO-PLANAR-SEAM-FUSION](../../backlog.md#apollo-planar-seam-fusion); parent [ATLAS-APOLLO-BEAT-THE-REFERENCES](../../backlog.md#atlas-apollo-beat-the-references)

## Context

After ADR 0053 the batched planar driver serves every power of two from
4096 through 524288, and it trails RustFFT by about 1.2 times in `f64` at
16384 and 65536 alike. The per-pass instrument (`pinned_sections`, extended
to 65536 and 262144) attributes the route on the pinned performance core:

| N | deinterleave | stage set 1 | transpose | stage set 2 | reinterleave |
|---|---|---|---|---|---|
| 16384 f32 | 25% | 25% | 6% | 30% | 14% |
| 65536 f32 | 17% | 33% | 8% | 30% | 12% |
| 65536 f64 | — | — | 5% | 39% | 9% |
| 262144 f64 | 21% | 25% | 4% | 33% | 17% |

The two stage sets are about 63% of the route; the three layout passes are
the rest, and two of them, the deinterleave into padded planes and the
reinterleave back into the caller's buffer, are pure movement that RustFFT
never performs because it stays interleaved.

## Rejected arm: L1 column blocking

The first hypothesis was that the stage sets stream the planes from L2 on
every pass (1 MiB of `f64` at 65536, 4 MiB at 262144) and would run faster
over one-cache-line column blocks kept in L1 across all passes. The
implementation was a block loop around each stage set with per-element
arithmetic unchanged. Measured with the same instrument, the stage sets
slowed by 1.4 to 2.4 times at every length from 4096 to 65536, and at
262144 `f32`, where the planes exceed L2, they did not move at all (1.09
and 1.00 times). The premise is falsified: the stage sets are bound by
instruction issue, about 4.4 cycles per fused radix-4 quad on four lanes
with roughly 36 vector operations each, not by plane bandwidth, and a short
inner loop only adds outer-loop overhead. The arm was reverted before any
commit; the evidence is `output/apollo-planar-seams/sections-blocked.txt`
beside the baseline attribution.

## Decision

Delete the two movement passes instead of accelerating them. The first
pass of the time-decimated stage set is the one pass that loads every
element exactly once, so it reads the caller's interleaved buffer directly:
two interleaved vector loads and one register deinterleave replace the two
plane loads, and the bit-reversed row map the deinterleave pass applied
becomes the source row index. The last pass of the frequency-decimated set
stores every element exactly once, so it writes the caller's buffer
directly: one register interleave and two interleaved stores replace the
two plane stores, with the reinterleave's row map on the destination. Both
kernels take the buffer as an optional `source` or `sink` beside the
existing twiddle `fold`, which is the same mechanism; the odd-power split
route passes neither, since its decimated input and combined output have
sinks of their own. The scalar deinterleave and the vectorized reinterleave
kernel are deleted with their only callers. The split's own decimation is
vectorized in the same shape (revision below): `deinterleave_pairs` splits
each register pair at complex granularity and the sub-lane unpack then
splits the reals, so both halves land in the plane column order with
natural stores and the scalar loop stays as the reference.

Per-element arithmetic and its order are unchanged, so results are bitwise
those of the unfused route.

## Verification

The batched suite (direct-transform agreement at 4 through 4096, `f32`
agreement, round trips, the plan cache), the four-step workspace suite
(exact impulse spectrum through the selected route at 4 through 262144 in
both precisions and directions), the RustFFT differential at 65536 and
262144 in both precisions, and the dimension-1d plan suite.

## Measurement and stop criterion

`pinned_sections` for attribution; `phastft_comparison`, `engine_census`
and `twiddless_comparison` for the verdict against the references on the
same pinned core. Accept when the whole-transform intervals at 16384, 65536
and 262144 sit below the ADR 0053 intervals; reject and restore the passes
otherwise.

## Result and decision

Accept. Evidence is `output/apollo-planar-seams/` with its manifest.

Whole-transform medians, apollo `f64` in microseconds, three runs after the
change against the two ADR 0053 runs before it on the same pinned core:

| N | census before | census after | rustfft after | warm before | warm after |
|---|---|---|---|---|---|
| 4096 | 9.4 / 9.6 | 8.2 / 8.3 / 9.0 | 7.7 to 8.3 | 9.4 | 9.0 to 9.2 |
| 16384 | 43.4 / 46.2 | 40.8 / 41.5 / 40.8 | 36.1 to 36.3 | 43.0 | 42.1 to 44.1 |
| 65536 | 214 / 217 | 210 / 204 / 205 | 168 to 172 | 204 | 201 to 212 |
| 262144 | 1808 / 2581 | 1440 / 1345 / 1538 | 1122 to 1565 | — | — |

The cache-flushed census improves 3 to 12% at 4096 through 65536 with the
after-intervals below the before-intervals in every pairing but one (the
9.0 third run at 4096); the warm clone-inclusive instrument is neutral,
within 4% either way. `f32` moves less than 2% in both instruments. The
262144 rows moved with the host (RustFFT moved with them) and are not
counted.

The per-pass instrument overstated the available gain because its baseline
was taken under a tree-mate's builds; its clean rerun after the change
reads the route's sections 21 to 49% lower in `f64` than that baseline,
which is consistent with the passes being gone but not a measure of the
whole transform, whose gain is the census figure above. Both runs are
retained with that caveat.

What the two arms establish together: the stage sets are instruction-issue
bound. Blocking them to L1 made them slower, and removing two full memory
passes bought a small fraction of those passes' attributed time because
the register shuffles that replaced them compete for the same issue slots.
The remaining 1.2 times to RustFFT in `f64` is the fused radix-4 pass's
instruction count per element, which a radix-8 pass would cut by a third
at the cost of sixteen live vector registers on AVX2; that is the next
item's design question, filed as
[APOLLO-PLANAR-RADIX-DEPTH](../../backlog.md#apollo-planar-radix-depth).

Limits: one host, one core class; contention recorded, not excluded; no
cross-machine claim.

## Revision

2026-09-08: changed Proposed to Accepted after the replicated census above.

2026-09-09: `boundary::DeinterleaveDecimatedRows` replaces the scalar
decimation on the odd-power split's entry
([APOLLO-PLANAR-SPLIT-VECTOR-DEINT](../../backlog.md#apollo-planar-split-vector-deint)):
`pinned_sections` `deint` halves in `f64` (2048 4.4k to 2.4k cycles, 32768
81k to 50k) and falls 3.6 times in `f32` (3.5k to 1.0k, 66k to 25k);
`rustfft_comparison` `f32` 2048 3.0 to 2.35 µs, `f64` 32768 level with
RustFFT at 77.5 against 78 to 82. The combine is now the split's largest
seam, and fusing both seams into the stage passes as this ADR did for the
square route is the next item.
