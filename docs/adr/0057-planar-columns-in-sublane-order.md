# ADR 0057: Plane columns in the dispatched backend's sub-lane order

- **Status:** Accepted
- **Date:** 2026-09-09
- **Class:** [minor] [perf]
- **Item:** [APOLLO-PLANAR-SEAM-LANE-ORDER](../../backlog.md#apollo-planar-seam-lane-order); parent [ATLAS-APOLLO-BEAT-THE-REFERENCES](../../backlog.md#atlas-apollo-beat-the-references)

## Context

ADR 0056's attribution put the three seam sweeps at two to three times
the cost of a plain sweep. In the source and sink listings each row's
interleave was two `vpermpd` on port 5 and two unpacks: the unpack alone
is the interleave within 128-bit sub-lanes, and the permute exists only to
put the lanes back in memory order. Nothing in the stage sets cares which
column a lane holds, since every column is an independent transform.

## Decision

The planes hold each aligned group of `lanes` columns in the order the
sub-lane deinterleave produces (`0, 2, 1, 3` for four `f64` lanes,
`0, 1, 4, 5, 2, 3, 6, 7` for eight `f32` lanes, `0, 4, 1, 5, 2, 6, 3, 7`
for eight `f64` lanes and `0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7,
14, 15` for sixteen `f32` lanes), so both seams are the sub-lane unpacks
alone. hermes-simd gained the pair
(`interleave_sublanes`, `deinterleave_sublanes`) and the sub-lane width
(`SimdPermute::SUBLANE_LANES`) in PR 161, native on AVX2 and AVX-512 and
the flat operation where the register is one sub-lane. In apollo,
`LaneOrder` is the one map between memory and plane columns, carried in
both directions (`column`, plane to memory, and `plane`, its inverse):
the fold planes are built through the forward map, the odd-power
decimation writes through the inverse, the combine reads through the
forward map. The transpose applies the forward map as a register
relabeling on one side of the in-register tile transpose and the inverse
on the other, so every load and store keeps its natural row. The AVX2
orders are involutions and the AVX-512 orders are not (revision below),
which is why the two maps are distinct.

Every planar kernel dispatches through `vectorize`, the stage sets'
selector, so the order the probe reports and the order the seams produce
are one backend's by construction; the fixed-width boundary dispatch and
`BOUNDARY_LANES` are deleted. Where the batch is narrower than a register
the vector path never runs and the order is the identity.

## Verification

The batched suite, the four-step workspace suite, the RustFFT differential
at 65536 and 262144, the dimension-1d plan suite, the crate's API tests
and the DFT oracle sweep: 604 tests passing, results bitwise those of
memory order. `lane_order` pins the documented orders at every width,
the inverse against the forward map, the constant tables against the
runtime map, the register relabeling of the transpose on a scalar model
of the tile at every width, and the dispatched deinterleave itself
against `LaneOrder`; the scalar transpose runs under the AVX-512 orders
on every host. The stage-set listings
(`output/apollo-planar-lane-order/asm_order_*_f64_avx2.s`) carry no
`vpermpd` or `vperm2f128`; the sink loop is two unpacks and two stores per
row.

## Measurement

Paired `pinned_sections` on the pinned performance core, main against the
lane alternating, five pairs at 65536 after the transpose relabeling
(`output/apollo-planar-lane-order/ab_sections_*`), thousands of cycles:

| precision | sweep | main (five pairs) | lane |
|---|---|---|---|
| f64 | t1 | 132 / 129 / 131 / 135 / 132 | 140 / 131 / 131 / 137 / 129 |
| f64 | f2 | 247 / 196 / 249 / 218 / 221 | 214 / 201 / 245 / 264 / 247 |
| f64 | total | 608 / 556 / 613 / 586 / 585 | 581 / 561 / 609 / 649 / 606 |
| f32 | transpose | 25 / 24 / 24 / 24 / 24 | 35 / 32 / 32 / 25 / 25 |
| f32 | f2 | 131 / 146 / 147 / 141 / 148 | 151 / 147 / 146 / 148 / 148 |
| f32 | total | 306 / 318 / 323 / 314 / 321 | 338 / 329 / 329 / 327 / 321 |

Neutral at 65536 in both precisions; the first three `f32` pairs carry the
transpose regression the relabeling removed (the eight-row tile lost a
third of its time when its rows were addressed through the order, the
four-row tile nothing). At 4096 and 16384 three pairs each read within the
run-to-run spread in both precisions. The whole-transform instruments
agree: warm `f64` 16384 33.9 / 33.2 against 33.4 / 33.5 µs, 65536
188.7 / 178.3 against 195.1 / 188.0, with the reference arm moving 1 to 2%
between the pairs.

The permute removal therefore moves nothing measurable: the seam sweeps
are not bound by their shuffles. What the pairs exposed instead is that
the sink sweep's cost is a placement lottery. With the probe's buffer
shifted by a few complex elements against the scratch planes
(`output/apollo-planar-lane-order/e5_off*.txt`), `f2` at 16384 `f64` reads
30k, 45k or 82k cycles per call for offsets of 64, 0 and 256 bytes
modulo the page, and `f32` 11k or 24k; the plain sweeps do not move.
The sixteen destination rows of a sink block lie a multiple of 4 KiB
apart, so their lines fill the same L1 sets and evict the tile between
the sweep's passes whenever those sets hold it; the source sweep reads
sixteen rows with the same spacing.

## Consequences

The lane order stays: it is the minimal seam, it deletes the dual-width
boundary dispatch, and it costs nothing. The next increment is the one
the lottery names: stage each seam block through a contiguous buffer so
the sixteen rows' lines spread over every set and the tile survives,
filed as
[APOLLO-PLANAR-SEAM-STAGING](../../backlog.md#apollo-planar-seam-staging);
prefetch and the compact fold follow it.

Limits: one host, one core class; a peer session's builds contaminated
several runs, which the plain sweep `t2` (64k `f64`, 33k `f32` at 65536)
identifies and which the pairs above exclude.

## Revision

2026-09-09: the first form carried one map and assumed an involution,
which holds at the AVX2 widths and not at the AVX-512 ones (`0, 4, 1, 5,
2, 6, 3, 7` cycles 1, 4, 2). Both transposes and the decimation's scalar
form then computed wrong answers on AVX-512 hosts, which the hosted
runner is on some runs and not others: the landing runs of PR 357
passed on one without it and the whole planar suite failed on main at
`beecdfad` and `f3cfb8c1`
([APOLLO-PLANAR-LANE-ORDER-INVERSE](../../backlog.md#apollo-planar-lane-order-inverse)).
`LaneOrder` now carries the inverse map, the transposes use one map per
side, and the tests run the AVX-512 orders on every host.
