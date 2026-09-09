# ADR 0056: Planar stage sweeps over L1-resident tiles

- **Status:** Accepted
- **Date:** 2026-09-09
- **Class:** [patch] [perf]
- **Item:** [APOLLO-PLANAR-STAGE-SWEEPS](../../backlog.md#apollo-planar-stage-sweeps); parent [ATLAS-APOLLO-BEAT-THE-REFERENCES](../../backlog.md#atlas-apollo-beat-the-references)

## Context

ADR 0055 left the planar stage sets bound by memory operations: a radix-4
pass costs the same whether it computes or not. Its attribution puts one
pass at 14.6 cycles per four-lane quad at 16384 and 65536 `f64` alike
(`stages1` at 65536: 239k cycles for four passes over 4096 quads). A quad
moves 8 loads and 8 stores of 32 bytes, so a pass carries four line fills
and four writebacks per quad across the L1 boundary; at 14.6 cycles that is
about 1.8 cycles per line transfer, the L1 to L2 rate under mixed fill and
writeback. The pass is bound by bytes crossing L1 per element-stage, which
is set by the number of passes, not by ports or instructions.

ADR 0054 rejected L1 blocking after measuring it 1.4 to 2.4 times slower.
That arm blocked one cache line of columns across every stage: each row set
then touched eight scattered lines per block, the row loop ran two vectors
between row-set setups, and no stream continued long enough for the
prefetchers to follow. The rejection binds to that shape.

## Decision

Each stage set runs in sweeps of `SWEEP_STAGES = 4` consecutive stages. The
`2^S` rows those stages mix form a tile no other row touches during them,
so a sweep runs its passes (radix-4 pairs, then a radix-2 when the count is
odd) over one tile before the next, with the tile's columns blocked to
`TILE_BYTES = 16 KiB` of both planes so a block stays in L1 between the
passes. Every element crosses L1 once per sweep instead of once per pass;
each pass keeps its butterfly, register budget, twiddle order and
per-element operation order, so results are bitwise those of the unswept
sets. The time-decimated set ascends from stage 2 and the frequency-
decimated set descends from `len`, each greedy: full sweeps first, then
the remainder. The four-step fold, the interleaved source and the
interleaved sink ride the same passes as before. `sweep.rs` states the
tile geometry; `Columns` carries the block through the row-set driver.

Both stage sets are now their sweep loops; the per-pass loop nests they
carried are deleted.

## Verification

The batched suite, the four-step workspace suite (exact impulse spectrum
through the selected route at 4 through 262144, both precisions and
directions), the RustFFT differential at 65536 and 262144, the
dimension-1d plan suite, the crate's API tests and the DFT oracle sweep:
600 tests, all passing on the change, unchanged in their assertions. The
sweep module's own tests pin the stage grouping, the row-set index
spreading and the block width.

## Measurement

`pinned_sections` on the pinned performance core, 200 calls per size,
three runs of the sweep against the last ADR 0055 attribution
(`output/apollo-planar-sweeps/planar_sections_sweep4*.txt` against
`output/apollo-planar-radix8/planar_sections_spec.txt`), in thousands of
cycles per call:

| N | precision | stages1 before | after (three runs) | stages2 before | after | total before | after |
|---|---|---|---|---|---|---|---|
| 16384 | f64 | 60 | 45 / 46 / 45 | 87 | 86 / 84 / 92 | 159 | 142 / 148 |
| 65536 | f64 | 239 | 211 / 194 / 197 | 367 | 340 / 338 / 333 | 654 | 598 / 578 |
| 262144 | f64 | 2287 | 2192 / 1957 / 1915 | 2400 | 2560 / 1822 / 1771 | 4938 | 5023 / 3947 |
| 16384 | f32 | 34 | 27 / 26 / 27 | 39 | 39 / 37 / 38 | 78 | 71 / 70 |
| 65536 | f32 | 129 | 104 / 96 / 100 | 180 | 182 / 173 / 173 | 340 | 310 / 297 |
| 262144 | f32 | 967 | 965 / 895 / 869 | 814 | 1043 / 835 / 814 | 1890 | 2119 / 1789 |

The instrument now also records each sweep beneath its stage set (`t1..t3`,
`f1..f3`). At 65536 `f64` the sweep carrying no seam (`t2`) costs 64k to
65k cycles for two passes over 4096 quads, 7.8 cycles per quad, against
14.6 before: the tile halves the pass cost as the model predicts, and the
remaining 24 loads and 8 stores per quad from L1 sit at the load-port
rate. The three seam sweeps do not follow: the interleaved source (`t1`)
130k, the fold (`f1`) 135k to 139k, the interleaved sink (`f2`) 197k to
198k. At 16384 the same shape holds (`t2` 16k against `t1` 30k, `f1` 26k,
`f2` 57k to 65k). The seams are now most of both stage sets.

Whole-transform medians in microseconds on the pinned core, two runs each
(`output/apollo-planar-sweeps/sweep*_engine_census.txt` and
`sweep*_twiddless_comparison.txt` against `spec2_*` in
`output/apollo-planar-radix8/`); the second census run's rows at 65536 and
262144 carry a reference-arm shift of 50 to 500% and are discarded:

| N | instrument | before | after | rustfft after | phastft after |
|---|---|---|---|---|---|
| 16384 | census f64 | 37.6 | 32.9 / 35.2 | 36.1 / 34.5 | 47.6 / 48.6 |
| 65536 | census f64 | 201.9 | 196.0 | 171.6 | 232.2 |
| 262144 | census f64 | 1596 | 1208 | 1212 | 1264 |
| 16384 | warm f64 | 39.7 | 33.5 / 34.2 | 34.1 / 34.2 | |
| 65536 | warm f64 | 204.7 | 186.1 / 182.2 | 171.2 / 170.2 | |
| 16384 | warm f32 | 20.2 | 16.7 / 17.5 | 17.8 / 17.6 | |
| 65536 | warm f32 | 103.2 | 98.4 / 94.4 | 115.0 / 111.6 | |
| 4096 | warm f64 | 9.1 | 8.9 / 8.6 | 7.7 / 8.3 | |

Apollo is level with RustFFT at 16384 `f64` in both instruments, leads it
in `f32` at 16384 and 65536, leads PhastFT everywhere measured, and trails
RustFFT by 1.07 to 1.15 times at 65536 `f64` and by up to 1.15 at 4096
`f64`, where the whole transform is cache resident and the seam sweeps
are 60% of the stage time (`t1` 8k and `f2` 11k against `t2` 2k).

## Rejected

- **8 KiB column blocks.** Measured once (`planar_sections_sweep4_8k.txt`):
  the plain sweeps gain 2 to 6% and the seam sweeps lose 13 to 60%, the
  sink sweep at 16384 `f32` from 39k to 63k cycles. The seam passes pay
  per block, so the block stays at 16 KiB.
- **Six stages per sweep.** Cuts a trip through the planes only at nine
  stages (262144), the one length whose planes exceed L2 and whose runs
  vary by 20% between replicates; not measured, re-opened with a quiet
  replicated census of that length.
- **Interleaved radix-8 rows** (the design ADR 0055 filed). A radix-8
  butterfly on interleaved AVX2 rows moves a third fewer bytes per
  element-stage than the planar pair but issues about 1.1 vector
  operations per element-stage against the pair's 0.63, since each
  complex multiply becomes two shuffles, a multiply and an alternating
  FMA over two samples. The sweep halves the bytes without touching the
  arithmetic, and with the plain pass at the L1 load-port rate the
  remaining cost is the seams, which the interleaved layout would not
  remove: its source and sink are the same buffer traffic, and its fold
  the same table. The item is closed on this analysis.

## Consequences

What the attribution now names, in cost order at 65536 `f64`: the sink
sweep (198k, of which the interleave is two `vpermpd` and two unpacks per
row per quad on port 5 and the rest is write traffic to the caller's
buffer), the fold sweep (135k, reading a twiddle table as large as the
data), the source sweep (130k, the mirror of the sink). Each is its own
increment on the board: the seam lane order that makes the in-lane unpack
the whole interleave, software prefetch on the three seam streams, and a
two-level fold table a fraction of the data's size.
