# ADR 0058: The planar sink staged through a contiguous block

- **Status:** Accepted
- **Date:** 2026-09-09
- **Class:** [patch] [perf]
- **Item:** [APOLLO-PLANAR-SEAM-STAGING](../../backlog.md#apollo-planar-seam-staging); parent [ATLAS-APOLLO-BEAT-THE-REFERENCES](../../backlog.md#atlas-apollo-beat-the-references)

## Context

ADR 0057 found the sink sweep's cost to be a placement lottery: 30k, 45k
or 82k cycles per call at 16384 `f64` as the caller's buffer moved 64, 0
or 256 bytes modulo the page against the scratch planes, and 11k or 24k
in `f32`. The sixteen rows the sink writes for one tile are `n` bytes
apart, a multiple of 4 KiB from 4096 up, so their lines fill the same L1
sets and, whenever those sets hold the tile, evict it between the sweep's
passes. The source sweep reads sixteen rows with the same spacing.

## Decision

The sink writes one tile block at a time into a contiguous staging block
of `STAGING_LEN` complexes at the end of the scratch (`scratch_len` grows
by one block, 16 KiB whatever the scalar), and the block's rows copy out
afterwards, one sequential row move each, once the tile is dead. The sink
pass addresses rows by their tile-local index; the row reversal moves to
the copy. The frequency-decimated set takes the odd remainder of its stage
count first, so the sink rides a full sixteen-row tile at nine stages
rather than the two-row tile that ends `[4, 4, 1]`, where staging has
nothing to spread and its copy is pure cost. Staging applies up to
`STAGED_SINK_MAX_LEN` (2^16); above it the sink stores directly through
its bit-reversed row table, since the measurement below turns against the
copy once the planes overflow L2.

The source keeps its direct loads and its bit-reversed row table in the
pass. Every staged form of it measured slower: one memcpy per row
serialises each row's sixteen lines (the source sweep at 65536 `f64` from
130k to 173k cycles), and rows moved abreast through vector loads alias
one another's stores through the load-store unit's page-offset check,
padded or not (260k to 330k); the direct loads read four rows abreast as
four streams and stay at 130k. The same abreast form cost the sink 290k
against the row copy's 180k: the row copy is a `memcpy` of full lines,
which the string-move path writes without a read for ownership.

Results are bitwise those of the direct sink: per-element operation order
is unchanged, and the pass grouping never changes results.

## Verification

The batched suite, the four-step workspace suite (scratch extents pinned
with the staging block), the RustFFT differential at 65536 and 262144,
the dimension-1d plan suite, the crate's API tests and the DFT oracle
sweep: 606 tests. A new direct-transform test holds the odd-power split
route at 2048 and 8192: while the source was staged, a seam that read its
staged block whenever stage 2 was present, rather than when a source
existed, escaped the impulse and round-trip checks (the output was a
consistent permutation, invisible to an impulse and undone by an
inverse) and surfaced only through the Bluestein-padded prime squares.

## Measurement

`pinned_sections` on the pinned performance core with the probe's buffer
shifted by `APOLLO_PROBE_OFFSET` complex elements (0, 16, 4: page offsets
of 0, 256 and 64 bytes), thousands of cycles per call
(`output/apollo-planar-sink-staging/stage7_off*.txt` against ADR 0057's
`e5_off*.txt` and the paired mains):

| precision | N | sweep | before (by offset) | after (by offset) |
|---|---|---|---|---|
| f64 | 16384 | f2 | 45 / 82 / 30 | 30 / 31 / 30 |
| f64 | 16384 | total | 129 / 172 / 116 | 111 / 114 / 114 |
| f64 | 65536 | f2 | 234 / 233 / 230 | 188 / 187 / 183 |
| f64 | 65536 | total | 605 / 603 / 594 | 555 / 546 / 547 |
| f32 | 16384 | f2 | 11 / 11 / 24 | 16 / 17 / 16 |
| f32 | 16384 | total | 66 / 65 / 72 | 58 / 60 / 62 |
| f32 | 65536 | f2 | 136 / 150 / 130 | 79 / 81 / 76 |
| f32 | 65536 | total | 312 / 326 / 299 | 248 / 253 / 247 |

The sink sweep no longer depends on placement (spread under 5% across the
offsets) and sits at or below the best offset's value everywhere but
`f32` 16384, where the best placement's 11k was the eight-row tile's
lucky case and 16k is its floor. The source sweep keeps a milder
placement dependence (`f32` 16384 `t1` 15k to 19k).

Whole transform, microseconds, main against the staged build alternating
(`st_*`): warm `f64` 4096 8.7 / 8.3 / 8.2 / 8.3 against 7.5 / 7.6 / 7.4 /
7.4 with RustFFT at 7.3 to 8.4; warm `f64` 65536 183.7 / 187.0 / 190.6 /
192.1 against 180.2 / 178.8 / 174.2 / 201.9 with RustFFT at 170 to 172;
warm `f32` 65536 90.6 / 91.7 / 100.8 / 91.5 against 99.8 / 91.9 / 75.5 /
75.7; warm `f64` 16384 33.2 / 32.7 / 33.0 / 34.1 against 37.1 / 35.0 /
33.9 / 33.4; warm `f32` 16384 16.3 / 20.4 / 16.2 / 16.2 against 19.0 /
18.1 / 18.3 / 18.1. Census `f64`: 4096 8.8 to 7.8 (RustFFT 8.2), 16384
37.7 to 33.8 (RustFFT 34.6), 65536 185.9 to 175.8 (RustFFT 171 to 172).
The warm `f32` 16384 rows read 12% slower while the attribution reads 10%
faster; the warm instrument clones its input each iteration, so its
placement is the allocator's, and the source sweep's remaining placement
dependence is the candidate, filed with the prefetch item.

At 262144, where the planes overflow L2, staging loses: with the sink on
a full tile the sink sweep read 1067k cycles against 800k direct in the
paired run, and the census read 1231 µs against 1173 with both reference
arms still (RustFFT 1230 / 1222, PhastFT 1300 / 1294). There the sink is
bound by its own writes, and one row copy after another loses the
four-row overlap of the direct stores. The sink is therefore staged up to
`STAGED_SINK_MAX_LEN = 2^16` and direct above it, a bound tied to this
host's 3 MiB L2 against the planes plus the caller's buffer. The census
pair after the gate (`st_main3` / `st_stage3`) reads 4096 8.9 to 8.0 and
65536 186.5 to 178.7 µs with RustFFT at 7.6 / 7.8 and 170 / 171; its
16384 and 262144 rows carry reference-arm shifts of 6 to 24% and are not
counted. Above the gate the sink is main's direct store, so 262144 keeps
main's behaviour apart from the sweep grouping.

## Consequences

Apollo leads RustFFT at 4096 and 16384 `f64` in the census and sits 1.02
to 1.05 behind at 65536 `f64`; `f32` leads at 65536 by 10 to 20%. The
next levers are the source sweep (prefetch, or a staged form that keeps
four streams in flight) and the fold table; both items stand on the
board.
