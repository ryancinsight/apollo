# ADR 0061: A two-pass base of 256 for the L1-resident powers of two

- **Status:** Accepted
- **Date:** 2026-09-11
- **Class:** [minor] [arch] [perf]
- **Item:** [APOLLO-L1-BASE-TWO-PASS](../../backlog.md#apollo-l1-base-two-pass); parent [ATLAS-APOLLO-BEAT-THE-REFERENCES](../../backlog.md#atlas-apollo-beat-the-references)

## Context

The powers of two from 512 to 4096 run through the base-128 route: a
gather into stride-`blocks` sub-transforms, one 128-point base per block
(eight 16-point rows in two radix-4 passes through a staging buffer, then
an eight-point column pass), the combine levels fused into the last
blocks' store sinks, and at 1024 one radix-2 level in place. Measured on
the pinned performance core with the phase meter
(`output/apollo-base128/split_attribution_2026-09-10.txt`, `f64`): 512 is
gather 205 + four blocks of (rows 235, columns with sink 311) + 365
unattributed = 2754 cycles against RustFFT 2258 (1.22); 1024 is gather 942
+ eight blocks of (235, 319) + 771 unattributed + one level 763 = 6908
against 5470 (1.29). The bare kernel is 65% of RustFFT's whole budget at
1024; the construction around it is the gap, and its two remaining levers
(the gather in the loads, the level in the second sink) are each bounded
at a few percent by aliasing between the blocks' sources and the sinks'
outputs ([gather in loads](../../backlog.md#apollo-base128-gather-in-loads)).

RustFFT 6.4.1 (`src/avx/avx_planner.rs`) plans 512 as one hand-written
butterfly and 1024 as four 256-point butterflies under one radix-4 step;
its `Butterfly256Avx64` is two passes, eight-point column butterflies
with twiddles and an in-register eight-way transpose into scratch, then
32-point rows. A 1024-point transform there touches the data four times;
ours six.

## Decision

Build a 256-point two-pass base over hermes registers as a second
instance of the base kernel's shape, and run 512 to 4096 as that base
under one radix-2 or radix-4 step, so the route makes four passes: the
step's decimation, the base's two, and the step's butterfly fused into the
base's store sink as the base-128 route already does. Keep the base-128
kernel for 128 and 256 while the new base is measured; retire it where the
new base wins.

Recommended shape (option B below, after the revision of 2026-09-11):
eight 32-point rows through the row phases widened from sixteen to
thirty-two samples, then the eight-point column pass unchanged. Option A
(sixteen 16-point rows, the column pass widened to sixteen) was built and
measured first and gained nothing over the split at 256. 512 is the same
kernel as sixteen 32-point rows in one pass — the sixteen-point column
network in registers, spilling — rather than two 256-blocks under the
step, at both widths (the revisions of 2026-09-11, the single-pass 512
base); the two-block form of the step is deleted, and the step serves
1024 as four blocks under the radix-4 sink.

## Options

- **A. A 256 base as 16 x 16 in the base-128 kernel's shape** (built,
  measured, rejected; see the revision). Reuses the two-pass structure,
  the staging discipline, the sinks and the plan tables. The sixteen-row
  column pass cannot hold sixteen columns in the AVX2 file, so its
  distance-8 level ran as one more pass over the 4 KB staging buffer, and
  that pass costs what the two eight-row column passes cost together.
- **B. A 256 base as 8 x 32, RustFFT's shape** (recommended). Eight-point
  columns keep the existing column pass and its register budget; the rows
  become 32-point, the row phases widened from `4 x 4` to `4 x 8` through
  the same pair staging (`dft32` exists as a scalar Winograd codelet, not
  as a register kernel). Risk: a 32-point row kernel over registers is new
  work with no sibling to copy, and the 8 x 32 twiddle layer is wider than
  16 x 16's.
- **C. Keep the base-128 route and take its two levers.** Bounded at 2 to
  5% each by the aliasing analysis; cannot reach parity (1.22 and 1.29).

## Consequences

- The split's gather kernel, the combine sinks and the level combine keep
  their roles above the new base; the base-128 kernel's phase structure is
  generalized over `ROWS` (already a const parameter) rather than cloned.
- The plan tables (`BasePlan<T, ROWS, TABLE_LANES>`) instantiate for
  `ROWS = 16`; the plan cache and the thread-local scratch grow by the
  larger staging buffer (4 KB at `f64`).
- `f32` follows through the wide kernel (`instance_major::wide`) once the
  `f64` base is level; until then `f32` keeps the base-128 route.

## Verification plan

1. The base alone: `transform_256` against the DFT oracle and the
   base-128 route bitwise-within-roundings on 256-point inputs, both
   directions, both precisions; the differential suite the base-128 kernel
   carries (`base128/tests.rs`) instantiated for the new base.
2. The base's cost: the phase meter on the new kernel and the pinned
   small-sizes probe at 256, 512, 1024 against RustFFT
   (`small_sizes_against_the_references_by_core_type`); acceptance is 512
   and 1024 not above RustFFT in `f64`.
3. The route: the dimension-1d executors selecting the new base for 512 to
   4096, the oracle suites, `rustfft_comparison` counterbalanced.

## Revision notes

- **2026-09-11, option A measured.** The sixteen-row form was built as
  the base kernel's `ROWS = 16` instance (`transform_256`; the column pass
  as one in-place distance-8 level over staging, four rows at a time, then
  the eight-row DIF once per half with the halves interleaved in the
  output) and wired at n = 256 in place of the two-block split
  (`output/apollo-base128/base256_sixteen_row_form_2026-09-11.patch`, 623
  tests green, the direct-transform oracle in both directions). On the
  pinned performance core, quiet host
  (`output/apollo-base128/small_sizes_base256_2026-09-11.txt`), `f64` 256
  reads 1.22 of RustFFT's 256 butterfly where the two-block split read
  1.20 the day before (`small_sizes_2026-09-10.txt`; the ratio is the
  stable quantity across runs, absolute counts drift by about 8%), and
  128 reads 1.02. The extra level pass over staging (sixteen loads,
  sixteen stores, eight butterflies and eight multiplies per group) costs
  about what both eight-row column passes cost, so the form is two passes
  and most of a third; at 256 the split is four passes too, and no pass
  is saved before 1024. The slice is reverted; the recommendation moves to
  option B, whose column pass keeps eight rows and whose extra work is in
  the row phases, where the kernel already stages through registers.
- **2026-09-11, option B measured.** The eight-row form over 32-sample
  rows (`ROW_LEN = 32`: radix-4 over `b1` within each stride-8 group, the
  `W_32^{b0 m}` layer from six broadcasts under rotations and signs,
  radix-8 over `b0` through the pair staging, the eight-point column pass
  unchanged) runs n = 256 oracle-correct in both directions and reads
  1.08 of RustFFT's 256 butterfly in two quiet pinned runs
  (`output/apollo-base128/small_sizes_rows32_2026-09-11.txt`, `_run2_`),
  against the split's 1.20 and option A's 1.22; the route at 256 gains
  10%. The base is the two-pass kernel the decision needs; the route above
  it (512 and 1024 as two and four 256-blocks under one radix step) is
  the next slice, and the acceptance stands on it.
- **2026-09-11, the route over the 256 base measured.** The split route
  generalized over its base (one `transform_via_base` over the row
  length, the base length and the block's lane count; the sinks and the
  gather over the block's lane count) runs 512 as two and 1024 as four
  256-blocks under one radix step. The phase meter at 1024, `f64`
  (`output/apollo-base128/small_sizes_route256_run{2,3}_2026-09-11.txt`):
  gather 1234 to 1312, four blocks of (rows 554, columns with the final
  sink 717 to 767), no level, 6783 to 6890 in all, against the 128 route's
  7324 to 8636 in the same runs (a 7 to 20% shorter construction); a
  256-block costs 8% more than two 128-blocks (1271 against 1180 cycles),
  its 4 x 8 rows 554 against 480. On wall clock the route reads 1.15 to
  1.26 at 512 and 1.25 to 1.33 at 1024 across three runs, where the 128
  route read 1.20 to 1.26 and 1.17 to 1.35; the gain is inside the run
  drift. The route lands on its meter and its consolidation (one generic
  split over the base, the 128 form its `f32` eight-lane instantiation).
  Parity at 512 and 1024 is not reached: what remains is the base
  kernel's per-sample cost (1.08 to 1.21 of RustFFT's 256 butterfly)
  and the gather and sinks around it, not the pass count. Status stays
  Proposed; the next slice attributes the 256 base against RustFFT's
  `Butterfly256Avx64` at the instruction level.
- **2026-09-11, the rows reordered on the census.** The asm census of the
  `f64` 256-block loop read 310 instructions and 45 spills a row pair for
  the `4 x 8` rows (the second stage held two radix-8 outputs for the pair
  transpose), against 197 and 11 for sixteen-sample rows. Reordered to
  `8 x 4` (radix-8 over `b1`, the layer, radix-4 over `b0`, the second
  stage the sixteen-sample form's over `ROW_LEN / 8` pairs): 234
  instructions and 20 spills. The rows' meter did not move (560 against
  554 cycles a block; `output/apollo-base128/small_sizes_rows8x4_run{1,2}_2026-09-11.txt`),
  so the spills were not the cost: the 32-sample row pays its layer,
  twenty-one general multiplies against the sixteen-sample row's nine.
  The order lands for its smaller loop and its shared second stage; the
  per-sample gap to RustFFT's butterfly stays the open question, with
  its own kernel's census the next reading.
- **2026-09-11, both kernels under one census.** RustFFT's
  `Butterfly256Avx64` at `f64` (its methods instantiate in the consumer's
  test binary; `output/apollo-base128/base256_2026-09-11.md`): the
  32-point row pass 424 instructions, 43 spills and 65 shuffles per column
  set, four sets; the eight-point column pass 109 instructions and no
  spills per set, sixteen sets; about 1700 and 1740 per 256 points.
  Apollo's `8 x 4` rows: about 750 instructions a row pair, 3000 per 256
  points; the column pass 83 per group, about 1330. The column pass is
  at or below the reference; the row phase runs about 1.8 times its
  instruction count, the excess being the `zbuf` round trip between the
  two radix stages and the pair transpose into staging, which RustFFT
  does not pay (its rows run in registers, spilling, and its transpose
  rides the column pass). The next slice holds the 32-sample row in
  registers without the intermediate plane, its outputs stored straight
  into staging through the pair transpose.
- **2026-09-11, the row pair in registers.** The row phase is its own
  module (`instance_major/rows.rs`): the four first-stage groups and the
  second-stage pairs are constant-indexed monomorphizations, the whole row
  pair live across the crossover, no `zbuf`. Census per `f64` row pair:
  532 instructions and 50 stack moves, straight-line, against the plane's
  about 770 and 73 (RustFFT 424 and 43). Meter: rows per 256-block 532 to
  536 cycles against 560 to 562, the block about 4% faster
  (`output/apollo-base128/small_sizes_rowsreg_run{1,2}_2026-09-11.txt`);
  wall clock inside run drift. A 31% instruction cut moving the meter 4.5%
  reads the row phase as latency-bound on its butterfly chains. Two
  findings the module documents: a closure or `array::from_fn` inside the
  kernel compiles outside the dispatcher's target-feature frame (its
  intrinsics become calls), and `Vector::zero()` re-probes the host, so the
  zero comes from the dispatch token. The larger gap is now `f32` at 256
  and 512 (1.29 to 1.38 of RustFFT), where the eight-lane width declines
  32-sample rows.
- **2026-09-11, the rows at eight lanes for every row length.** The row
  module is generic over the register's sample count (`S = 2` at four
  lanes, `S = 4` at eight): one first stage, the layer's broadcasts a
  strategy (the four-lane dup-split table chunks, or eight-lane register
  pairs splatted once per transform), the `S x S` sample transpose the
  only other difference. The eight-lane kernel (`wide.rs`, sixteen-sample
  rows over its own `zbuf`) is deleted and the plan no longer declines
  32-sample rows on eight-lane hosts. Census at eight lanes: 594
  instructions and 101 stack moves a four-row 32-sample group; `f64`
  unchanged. Pinned probe, `f32` of RustFFT: 256 1.07 to 1.10 (from 1.29
  to 1.32), 512 1.18 to 1.24 (from 1.33 to 1.38), 128 and 1024 about
  1.07 (`output/apollo-base128/small_sizes_{wide32,aligned}_run*_2026-09-11.txt`).
  The same runs moved `f64` 1024 wall clock 10% with its meter unchanged:
  the probe's `Vec` buffers sit at 16 bytes and the new `f32` plans
  shifted the heap, so the probe now works every arm in a 64-byte-aligned
  range; aligned, `f64` reads 256 1.11, 512 1.16 to 1.21, 1024 1.33 to
  1.34 (RustFFT itself had been misaligned before). Remaining levers in
  order: the 1024 gather (15 to 19% of the route), the column phase, and
  the row chain depth.
- **2026-09-11, the blocks read the parent.** The kernel takes a source
  and an output surface: the source is the block itself or block `OFFSET`
  of a `BLOCKS`-way split read at stride out of a parent, and the sinks
  index the surface holding no `&mut` of their own, so a combining block
  reads the parent it then writes (the rows finish before the first
  store). At four lanes the gather pass is gone — at 1024 subsequences 0
  and 1 run into scratch as the pairs' peers, 2 runs over the parent into
  the free scratch pair, 3 over the parent with the final sink — and `f64`
  1024 reads 1.15 to 1.23 of RustFFT from 1.33 (1.34 to 1.41 ms from
  1.53). At eight lanes the same loads cost two or four windows and up to
  three shuffles a register, more than the gather (`f32` 512 and 1024 2 to
  4% slower), so that width keeps the two- and four-block gather and the
  even half in the output through an in-place final sink; the plan width
  selects the form. The base-128 split, unreachable since the 256 base
  exists on every host the 128 one does, is deleted with its eight-block
  arm, its level pass, and the probe's incumbent comparison
  (`output/apollo-base128/base256_2026-09-11.md`).
- **2026-09-11, the sink twiddles dup-split in the base state.** The
  base plan state owns the split's sink tables for its route length
  (`SplitSinks`: `W_{2 BASE}^j` and `W_{4 BASE}^j` in the dup-split
  chunk-pair layout at the plan's register width, per direction on first
  use, relaid from the process twiddle cache), so a sink's complex
  multiply is one swap, one multiply, and one `fmaddsub` where the
  interleaved form paid three shuffles; the plan keeps no interleaved
  table for the split. Pinned probe: `f64` 512 1.12 to 1.13 of RustFFT
  from 1.20 to 1.25 (514 us from 556 to 566), 1024 1.14 to 1.18 from
  1.22 to 1.23; `f32` 512 288 to 290 us from 295 to 296, 1024 645 to 658
  from 655 to 662. The meter's 1024 column phase reads higher (668 to 714
  a block from 630) while the production wall clock improved: the doubled
  tables take the 1024 working set past the 48 KB L1D, the hypothesis the
  next slice tests (`output/apollo-base128/small_sizes_sinktables_run{1,2}_2026-09-11.txt`).
- **2026-09-11, the outer sink table interleaved.** Tested and kept: the
  outer level's table stays interleaved and the final sink duplicates
  each twiddle in registers, the inner table dup-split. Meter at 1024
  5227 to 5802 from 6011 to 6075 (columns 563 to 654 a block from 668 to
  714), wall clock `f64` 1024 1.36 ms from 1.37 to 1.38, 512 and `f32`
  unchanged (`output/apollo-base128/small_sizes_outerint_run{1,2}_2026-09-11.txt`).
  After slices 5 to 8 the `f64` route reads 512 at 1.09 to 1.13 and 1024
  at 1.14 to 1.21 of RustFFT, from 1.17 to 1.19 and 1.26 to 1.27 when
  the item opened; the row phase, latency-bound on its butterfly chains,
  is the lever left.
- **2026-09-11, the layer pre-rotated.** A critical-path estimate over
  the row loop's asm (`output/apollo-base128/chain.py`) put the `f64` row
  pair between its chain (about 114 cycles) and its issue bound (132 to
  176), against RustFFT's 77 and 106 to 141 for 424 instructions: the
  count and the chain both matter, and the excess is shuffle-class. The
  32-sample layer's broadcasts are pre-rotated (twelve instead of six) so
  no rotation or sign follows a layer multiply: 511 instructions and 114
  shuffles a row pair from 532 and 122; the 256 base 2.5% faster at both
  scalars on the pinned probe, the split lengths inside drift
  (`output/apollo-base128/small_sizes_prerot_run{1,2}_2026-09-11.txt`).
- **2026-09-11, the four-block step as one radix-4 sink.** The
  four-block split had combined in two levels, writing the even half to
  a pair of blocks and reading it back for the outer level.
  `FinalRadix4Sink` forms both halves from the three transformed spectra
  and applies the outer level as the last block's registers leave the
  kernel, so the first three blocks run into scratch through the direct
  sink at either width and no intermediate pair exists. Pinned probe:
  `f64` 1024 1.11 to 1.12 of RustFFT from 1.22 to 1.25 (1.28 ms from
  1.38 to 1.43), `f32` 1024 1.04 to 1.07 from 1.08 to 1.09, 512 1.16 to
  1.19 from 1.15 to 1.23
  (`output/apollo-base128/small_sizes_radix4_run{1,2}_2026-09-11.txt`).
  After slices 5 to 10 the `f64` route reads 512 at 1.12 and 1024 at 1.11
  to 1.12 of RustFFT, from 1.17 to 1.19 and 1.26 to 1.27 when the item
  opened; the row phase remains the lever.
- **2026-09-11, the column pass's registers named.** The chain estimate
  put one column group at about 67 cycles of critical path for 20 of
  issue (RustFFT's eight-point set 57 for 26): the column phase is a
  chain problem living on the out-of-order window, about two groups in
  flight on the meter. The group as named bindings with constant indices
  and dup-split `W_8^{1,3}` pairs is instruction-neutral (84 and 15
  shuffles from 83 and 17, the eight stack moves unchanged) and kept for
  the form; the wall clock read inside drift
  (`output/apollo-base128/small_sizes_colreg_run{1,2}_2026-09-11.txt`).
- **2026-09-11, two column groups interleaved (rejected).** Two eight-row
  groups an iteration with their loads, twiddles, and stages interleaved
  at the source: the meter's 1024 column phase 613 to 628 a block from
  520 to 527, `f32` 1024 638 to 640 us from 611 to 612, the 256 base
  218 us from 209 to 210 — sixteen live columns spill past what the
  interleave buys, and the out-of-order window overlaps groups better
  than the source can
  (`output/apollo-base128/small_sizes_colpair_run{1,2}_2026-09-11.txt`).
  Reverted. With that, the base shape is decided and this record moves
  to Accepted: the 256-point instance-major base with 32-sample rows in
  registers at either width, the split over it reading the parent at
  four lanes and gathered at eight, the radix step as the last block's
  sink, the sink twiddles owned by the plan state. The route reads `f64`
  512 at 1.08 to 1.15 and 1024 at 1.13 to 1.17 of RustFFT, `f32` 512 at
  1.16 to 1.19 and 1024 at 1.03 to 1.08, from 1.17 to 1.38 when the item
  opened; what remains is inside the kernels — the row pair's 511
  instructions against RustFFT's 424 (the pair transpose 32 of them, the
  eighths 12) and the column group's 67-cycle chain — and is the
  follow-up item's, not a shape decision.
- **2026-09-11, the eighths as dup-split multiplies.** The radix-8 is
  generic over an `Eighths` strategy: `HalfRoot2` keeps the
  `sqrt(2)/2` form (a rotation, a butterfly arm, a real multiply — four
  operations, a chain of twelve) for the Stockham and Winograd kernels,
  and the base rows take `DupSplitEighths` from two more plan broadcasts
  (three operations, a chain of nine). Census, `f64` row pair: 493
  instructions from 511, chain 127 from 147. Pinned probe: rows per
  256-block 532 to 549 cycles from 578 to 588; `f64` 256 1.00 to 1.01 of
  RustFFT from 1.05 to 1.06, 512 1.00 to 1.08 from 1.11 to 1.15, 1024
  1.04 to 1.07 from 1.13 to 1.17; `f32` 256 1.03 to 1.04, 512 1.06 to
  1.10, 1024 1.03 to 1.04
  (`output/apollo-base128/small_sizes_eighths_run{1,2}_2026-09-11.txt`).
- **2026-09-11, the outer level's high twiddles by rotation.** The
  radix-4 sink forms `W_{4 BASE}^{j + BASE}` as the quarter turn of
  `W_{4 BASE}^j`, so one outer table serves both halves (4 KB at `f64`
  1024 from 8). The meter reads the 1024 column phase at 571 to 589
  cycles a block from 634 to 656; the wall clock inside drift
  (`output/apollo-base128/small_sizes_outerrot_run{1,2}_2026-09-11.txt`).
- **2026-09-11, the two-block split strided at eight lanes (rejected
  again).** Re-measured after the eighths and the rotated outer table:
  the eight-lane two-block route reading the parent runs 612
  instructions a row group against 543 gathered, and `f32` 512 reads
  272 to 273 us against 271 to 278 — the gather pass and the in-register
  extraction cost the same at this width, so the gathered form stays
  (`output/apollo-base128/small_sizes_f32strided_run{1,2}_2026-09-11.txt`).
  The route after slices 5 to 14: `f64` 128 1.02 to 1.07, 256 0.99 to
  1.02, 512 0.98 to 1.08, 1024 1.00 to 1.07 of RustFFT; `f32` 128 1.03
  to 1.10, 256 1.00 to 1.04, 512 1.06 to 1.14, 1024 0.96 to 1.05.
- **2026-09-11, the single-pass 512 base at eight lanes
  (`APOLLO-SINGLE-PASS-512-BASE`).** The kernel at `ROWS = 16` over
  32-sample rows, one block of 512: the row phases unchanged (540
  instructions a four-row group, four groups), the column pass a
  sixteen-point DIF in registers — one distance-8 stage under `W_16^a`
  (the odd multiples as dup-split broadcasts, `W_16^{2,6}` the eighths,
  `W_16^4` the quarter turn), then the eight-point network on each half,
  the low half's register `q` stored to row `2 rev3(q)`, the high half's
  to `2 rev3(q) + 1`. Option A's objection stands as a fact — sixteen
  columns and their broadcasts exceed the AVX2 file, and the group spills
  (223 instructions, 35 stack moves, a chain of about 104 cycles for 54
  of issue, over eight groups) — but the spills are not the cost: the
  instruction count is 5% above the two-block route's (3944 against
  about 3760 with its gather), and the wall clock is 12% below it. What
  the single pass removes is the two-block route's passes — the gather,
  the second block's round trip through scratch, the combining sink's
  table reads — not instructions. Pinned probe, two runs
  (`output/apollo-base128/small_sizes_base512_run{1,2}_2026-09-11.txt`):
  `f32` 512 239 / 237 us against RustFFT's 238 / 237 (1.01 / 1.00, from
  1.10 to 1.12 on the two-block route in the strided runs), on the
  efficiency core 491 / 491 against 504 / 504 (0.97 from 1.20); every
  other length inside its drift (`f32` 256 1.02 / 1.00, 1024 1.02 /
  1.04; `f64` unchanged, the sixteen-row form declining at four lanes).
  The direct-transform oracle is green at 512 in both directions at
  `f32`; `f64` asserts the decline. Selected at n = 512 where the plan is
  eight-lane; the four-lane width keeps the two-block split, its own
  sixteen-row form (8 KB of staging at `f64`) unmeasured and out of the
  item's scope.
- **2026-09-11, the sixteen-row form at four lanes
  (`APOLLO-SINGLE-PASS-512-FOUR-LANES`).** The restriction lifted and
  measured: the four-lane two-block route already read the parent, so
  what the single pass removes there is the second block's round trip
  through scratch and the combining sink's table reads. Pinned probe,
  two runs
  (`output/apollo-base128/small_sizes_base512four_run{1,2}_2026-09-11.txt`):
  `f64` 512 434 / 441 us against RustFFT's 461 / 452 (0.94 / 0.98, from
  1.03 to 1.05 on the two-block route in the base512 runs), on the
  efficiency core 1013 / 1008 against 1045 / 1043 (0.97, from 1.11);
  `f32` 512 unchanged at 0.97, every other length inside its drift. The
  census reads the four-lane sixteen-row column group at 222
  instructions and 34 stack moves (chain about 104 for 54 of issue) over
  sixteen groups, the row group 493 over eight — the same shape as at
  eight lanes. Kept at both widths; the two-block form of the radix step
  (`two_blocks_direct`, `two_blocks_gathered`, `CombineSink`, the
  two-block gather and strided arms) is deleted as superseded, the step
  serving 1024 as four blocks under the radix-4 sink. The route now
  reads `f64` 128 1.03 to 1.05, 256 0.98 to 1.00, 512 0.94 to 0.98, 1024
  1.03 to 1.05 of RustFFT; `f32` 128 1.04 to 1.10, 256 1.01, 512 0.97,
  1024 0.99 to 1.02.
- **2026-09-11, 1024 as two sixteen-row blocks (rejected;
  `APOLLO-1024-AS-TWO-512-BLOCKS`).** The radix step generic over the
  base's row count, 1024 as two 512-blocks under the combining sink over
  `W_1024^j` — one block through scratch (8 KB at `f64`) against three
  (12 KB), the outer table gone, the parent read directly at four lanes
  and gathered at eight (`output/apollo-base128/two512_route_2026-09-11.patch`).
  Pinned probe, two runs
  (`output/apollo-base128/small_sizes_two512_run{1,2}_2026-09-11.txt`):
  `f64` 1024 1267 / 1241 us against RustFFT's 1153 / 1106 (1.10 / 1.12,
  from 1.03 to 1.05 on four blocks), `f32` 1024 644 / 633 against 575 /
  582 (1.12 / 1.09, from 0.99 to 1.02); the efficiency core `f64` 1.06 /
  1.08 (from 1.07) and `f32` 1.04 / 1.05 (from 1.07 to 1.10). The phase
  meter reads the two-block route at 4561 to 5137 cycles a call against
  the four-block route's 4794 to 5105 in the same runs — the meter and
  the wall clock disagree again (as in the route work of 2026-09-11), and
  the wall clock decides. What the two-block form saves in passes it
  spends in the sixteen-point column: two 222-instruction groups a column
  set against two 84-instruction ones and the radix-4 sink's level, and
  at 1024 the 256-block's passes were already few (no gather at four
  lanes, one radix-4 sink). The four-block route over the 256 base stays
  for 1024; the two-block form of the step stays deleted.

- **2026-09-11, the 128 base as four 32-sample rows
  (`APOLLO-128-AS-FOUR-ROWS`).** The census read the 8 x 16 kernel
  latency-bound in its column pass (84 instructions a group at a chain
  of about 67 cycles for 20 of issue, RustFFT's `Butterfly128Avx` loops
  at 1.3 to 1.9 times issue) with two (`f32`) or four (`f64`) row groups
  to overlap. The same kernel at `ROWS = 4` over 32-sample rows runs one
  or two 32-sample row groups and eight or sixteen four-row column groups
  (two levels, one rotation, no multiply). Measured as bare arms in one
  probe run beside the 8 x 16 form, two runs
  (`output/apollo-base128/small_sizes_128ab_run{1,2}_2026-09-11.txt`):
  at eight lanes (`f32`) 49.8 / 50.3 us against 53.1 / 53.8 (6% faster;
  RustFFT 49.6 / 47.9), at four lanes (`f64`) 89.1 / 88.4 against 85.8 /
  86.9 (2 to 4% slower; RustFFT 84.2 / 84.8); the efficiency core reads
  the two forms equal at both scalars. The 128 state is an enum over
  the two shapes, selected by the native width once at plan build. The
  phase meter read the four-row form 19% under the 8 x 16 one at `f64`
  while the wall clock read it slower — the third meter-against-clock
  divergence of the day, all decided by the clock. The plan-level probe
  arm at `f32` 128 read 57.9 / 64.8 us against the bare arm's 50.3 /
  49.8 in the same runs, while earlier runs read the two equal: the
  same kernel through a different table allocation, the alignment
  question the work buffers were already cured of, filed as its own
  item. Re-measured under aligned tables (the next revision), two runs:
  4 x 32 49.9 / 50.3 us against 8 x 16 51.7 / 51.6 at `f32` (3% faster),
  88.1 / 89.2 against 87.3 / 87.5 at `f64` (1 to 2% slower) — the
  width-selected shape at its true margin.
- **2026-09-11, the base tables and staging on the cache line
  (`APOLLO-BASE-TABLE-ALIGNMENT`).** The plan's dup-split table was a
  `Box<[T; N]>` and the split's sink tables `Box<[T]>`, at the
  allocator's 16 bytes, and the staging buffer a stack array at the
  scalar's alignment, so whether a 32-byte vector load split a cache line
  was the heap's or the frame's luck. The pinned probe, carrying both
  128 shapes as bare arms beside the plan-level arm, read the same
  kernel 8 to 30% apart through two allocations of one table
  (`f32` 128 plan 57.9 / 64.8 us against bare 50.3 / 49.8; `f64` 96.4
  against 88.4), and the bare arms of the two shapes reversed between
  runs. With the table in a 64-byte aligned wrapper, the sink tables
  starting at the first boundary inside their buffer, and the staging
  buffer in the same wrapper, two runs
  (`output/apollo-base128/small_sizes_aligned_run{1,2}_2026-09-11.txt`)
  read the plan-level and bare 128 arms within 1% (`f64` 87.3 / 88.0
  against 87.3 / 87.5, `f32` 51.9 / 51.9 against 51.7 / 51.6), and
  apollo's arms within 1.5% of themselves across the runs at every
  length from 128 to 1024 where the earlier runs moved up to 8%:
  `f64` 128 87.3 / 88.0, 256 200.5 / 197.4, 512 441 / 447, 1024 1190 /
  1188; `f32` 128 51.9 / 51.9, 256 109.7 / 109.3, 512 236.4 / 236.0,
  1024 596.1 / 596.8. RustFFT's arms, whose allocations are its own,
  now carry the drift (`f64` 128 100.2 / 86.6, `f32` 128 50.6 / 60.6),
  so a ratio against it is read from its better run. The run-to-run
  band the earlier revisions called the probe's drift was, in this
  measure, the tables' placement.

- **2026-09-11, 2048 as four sixteen-row blocks
  (`APOLLO-2048-AS-FOUR-512-BLOCKS`).** The scoreboard's widest gap was
  past the base route: 2048 on the generic power-of-two route read
  `f64` 1.38 to 1.40 of RustFFT and PhastFT and `f32` 1.62 of RustFFT.
  The radix step over the base is generic over the base's row count,
  so the 512 base serves 2048 as four blocks under the radix-4 sink
  with no new kernel (the parent read directly at four lanes, gathered
  at eight, three blocks through scratch: 24 KB at `f64`, 12 KB at
  `f32`), the plan taking it ahead of the four-step route. Pinned probe,
  two runs
  (`output/apollo-base128/small_sizes_2048_run{1,2}_2026-09-11.txt`):
  `f64` 2048 3459 / 3204 us against RustFFT's 2795 / 2898 (1.24 / 1.11),
  `f32` 1693 / 1686 against 1310 / 1307 (1.29); the efficiency core
  `f64` 1.19 / 1.18, `f32` 1.07 / 1.05. Kept: below the generic route at
  both scalars, and the remainder is the route's own — the meter reads
  each 512-block's column-and-sink phase at about 1900 cycles against
  1150 for its rows, the radix-4 sink over four 512-spectra and the
  outer `W_2048` table being the cost the 1024 route pays at half the
  size; the 4096 length (1.07 to 1.22) is the next step of the same
  ladder, eight or sixteen blocks under a deeper sink.

- **2026-09-11, 384 as three 128-blocks under a radix-3 step
  (`APOLLO-384-AS-THREE-128-BLOCKS`, slice 1).** The composite census
  (`output/apollo-base128/composite_census_2026-09-11.md`) read the
  composite route's butterflies at RustFFT's cost per register and its
  pass count at five to their two: RustFFT runs 384 as its 128-point
  butterfly under one 3xn column pass. The radix step over the base
  takes three blocks: a three-block parent source at four lanes (two
  windows and one interleave a register, the second window two samples
  on so the last chunk stays inside the parent), a radix-3 sink over two
  spectra and `W_384^j`, `W_384^{2 j}` (computed for the state, the
  stage-major cache serving powers of two only), the 128 state built at
  384 where its shape reads the parent at four lanes. Pinned probe, two
  runs (`output/apollo-base128/small_sizes_384_run{1,2}_2026-09-11.txt`):
  `f64` 384 345 / 341 us against RustFFT's 345 / 346 (1.00 / 0.99, from
  1.26), the efficiency core 827 / 826 against 739 / 740 (1.12, from
  1.29); `f32` unchanged at 1.37, the eight-lane shape declining the
  three-block source. Kept.
- **2026-09-11, the three-block source at eight lanes (slice 2).** The
  stride-three register at eight lanes from the four windows at
  `12 c + OFFSET` and three, six, nine samples on — column 0 of their
  transpose, the network the four-block source runs with its windows
  four apart — the last chunk's windows stepping back three samples and
  taking column 3, since the parent ends inside its window nine on.
  Both shapes of the 128 state serve 384. Pinned probe, two runs
  (`output/apollo-base128/small_sizes_384wide_run{1,2}_2026-09-11.txt`):
  `f32` 384 222 / 223 us against RustFFT's 254 / 188 (0.87 / 1.19, from
  1.37; 259 / 258 on the composite route), the efficiency core 501 / 500
  against 393 / 396 (1.28 / 1.26, from 1.33); `f64` 343 / 343 against
  465 / 346 (0.99 on its steadier run). Kept: below the composite route
  at both scalars. The `f32` residue (1.19 on the performance core, 1.26
  on the efficiency core) is the three-block route's own — four windows
  and three shuffles a register against the gather's two — and the
  four-block form's gather at eight lanes suggests the same trade here;
  a stride-three gather is the reading left.

- **2026-09-11, 4096 as eight sixteen-row blocks (rejected;
  `APOLLO-4096-AS-EIGHT-512-BLOCKS`).** The radix step at eight blocks:
  an eight-way gather at eight lanes, the parent read at stride eight at
  four, and a radix-8 sink over seven spectra deriving `W_4096^{j k}` for
  `j = 2..7` from the table's `W_4096^k` by multiplication, the register
  radix-8 across the blocks
  (`output/apollo-base128/eight_blocks_4096_route_2026-09-11.patch`).
  Pinned probe, two runs
  (`output/apollo-base128/small_sizes_4096_run{1,2}_2026-09-11.txt`):
  `f64` 4096 7239 / 7522 us against the four-step route's 7850 / 8029
  on the performance core (5 to 9% under it; RustFFT 8433 / 6453,
  PhastFT 7078 / 6429), 13958 / 14004 against 12587 / 12721 on the
  efficiency core (10% over); `f32` 4099 / 4183 against 3916 / 3949 (4
  to 7% over) and 7635 / 7732 against 7171 / 7173 (7% over). The working
  set is the reading: the parent and seven scratch blocks put 60 KB at
  `f32` and 120 KB at `f64` in flight, past L1 at both scalars where
  the 512 and 2048 forms stayed inside it, and the sink's six derived
  powers a chunk sit on the critical path. The four-step route stays for
  4096; a base route there needs an L2-aware shape (a 1024-point base
  under a radix-4 step, or blocks that finish inside L1 before the step),
  filed as the reading left.

- **2026-09-11, the per-call fixed cost of the lengths under 64
  (`APOLLO-SMALL-LENGTH-FIXED-COST`).** Attribution by asm census of
  the release test binary
  (`output/apollo-base128/base256_2026-09-11.md`): the `f32` eight-point
  call path was an executor of two instructions tail-jumping into the
  trait hop, which probed the host per call and called the register arm
  (63 instructions) — three frames for a 3 ns transform, against
  RustFFT's virtual call into one frame (5 + 43); the `f32` 32 executor
  carried two feature-detect calls and hermes' lane dispatch in front
  of the codelet. Two slices. The reduced eight-point arm in 256-bit
  registers (two registers, one fused multiply-add against `W_8`, the
  pair layout running both four-point transforms at once): `f32` 8 on
  the performance core 1.21 from 1.52, the efficiency core 1.15 to 1.21
  from 1.11 — the cross-lane permutes cost there. Then the vector frame
  entered once: the plan selects a framed executor set at construction
  when the host has AVX2 and FMA (hermes' `Avx2` contract), each
  executor carrying the frame so the sized entry and its register arm
  inline into it, and the `f32` 16 and 32 executors holding the hermes
  eight-lane codelets through the backend token. After: the `f32`
  eight-point executor is one frame of 29 instructions with no calls
  (RustFFT: a 5-instruction entry and a 27-instruction body), the 16
  and 32 executors 71 and 164 against RustFFT's 52 and 119 bodies.
  Pinned probe, three runs
  (`output/apollo-base128/small_sizes_frame_run{1,2,3}_2026-09-11.txt`):
  `f32` 8 1.07 / 1.07 / 1.07 on the performance core and 0.93 / 0.93 /
  0.94 on the efficiency core; no length under 128 slower, `f64` 8
  inside its eleven-run band. Kept. The remaining small-length gaps are
  the kernels, not the chain — `f32` 16 on the efficiency core 1.27 and
  `f32` 32 on the performance core 1.15 to 1.23 with the frame in
  place, the hermes primitives spending a rotation as four
  instructions (swap, add and subtract against zero, blend) against
  RustFFT's two (sign mask, swap) and a twiddle multiply as six against
  four (a sign flip and blend where RustFFT's `fmaddsub` folds the
  signs) — filed as `APOLLO-F32-16-32-KERNEL-GAP`.

- **2026-09-11, the rotation and the twiddle rows
  (`APOLLO-F32-16-32-KERNEL-GAP`, slice 1).** Two forms, one upstream
  and one here. hermes' `ComplexReg::mul_i` / `mul_neg_i` spent the
  quarter turn as an alternating fused multiply-add against zero, which
  LLVM expands to an add and a subtract against zero and a blend (it
  cannot fold `0 - c` to `-c` under signed zeros): four instructions per
  rotation in every radix-4 stage of every register kernel; now the swap
  and one xor against a sign-mask pair (hermes PR #172). And this
  crate's DFT-16 and DFT-32 twiddle rows were compile-time constants,
  whose negated half LLVM folded into the complex multiply's second
  constant and so broke the `fmaddsub` pattern into a sign flip, a blend
  and a plain `fmadd`; the rows are promoted constants loaded through
  `black_box`, one load a row, and the fused form returns. Census: the
  `f32` 16 body 68 vector instructions on a 33-cycle estimated chain
  (from 51), the 32 body 163 on 55 (from 66), against RustFFT's 48 / 34
  and 115 / 56. Pinned probe, two runs
  (`output/apollo-base128/small_sizes_kernelgap_run{1,2}_2026-09-11.txt`),
  apollo / RustFFT: `f32` 16 on the efficiency core 1.04 / 1.04 (from
  1.27), `f32` 32 on the performance core 1.01 / 1.11 (from 1.15 to
  1.23) — and the rotation form reaches every register kernel, so the
  base routes moved with it: 64 through 512 at both scalars and both
  cores 8 to 13% under their frame-run readings, `f64` 512 0.81 / 0.85
  on the performance core and 0.88 on the efficiency core, `f32` 256
  0.88 / 0.90, `f64` 128 0.89 / 0.92 (RustFFT's readings unchanged
  between the run pairs). The residue at 16 and 32 is the row loads'
  alignment checks (`cast_slice` on an opaque pointer, three at 16 and
  six at 32, each a compare, a branch and a trap) and the swap of each
  row for the multiply's second operand, where RustFFT holds both rows;
  slice 2.

- **2026-09-11, the twiddle rows as direct and swapped lanes
  (`APOLLO-F32-16-32-KERNEL-GAP`, slice 2; closed).** Each DFT-16 and
  DFT-32 row is a promoted constant of eight interleaved lanes in its
  direct form and with every sample's real and imaginary lanes
  exchanged, loaded opaque and multiplied through hermes'
  `ComplexReg::mul_with_swapped` (hermes PR #173): one load a row where
  the multiply had swapped the twiddle per use, and no slice cast, whose
  alignment check on an opaque pointer had cost a compare, a branch and
  a trap per row. Census: the `f32` 16 body 57 vector instructions
  (from 68; RustFFT 48), the 32 body 133 (from 163; RustFFT 115), no
  checks left. Pinned probe, two runs
  (`output/apollo-base128/small_sizes_kernelgap2_run{1,2}_2026-09-11.txt`),
  apollo / RustFFT: `f32` 16 on the efficiency core 0.97 / 0.96 (1.27
  before the item), `f32` 32 on the performance core 0.95 / 0.96 (1.15
  to 1.23 before), `f32` 32 on the efficiency core 0.77, `f32` 16 on
  the performance core 1.01 / 0.99; 64 through 512 hold their slice-1
  readings. Kept, and the item closed on its acceptance. What remains
  above RustFFT among the small lengths is `f64` 8 on the performance
  core (1.11 to 1.22, the scalar codelet against `Butterfly8Avx64`) and
  `f64` 32 on the efficiency core (1.04 to 1.07).

- **2026-09-11, 2048 as eight 256-blocks under a radix-8 step at eight
  lanes (`APOLLO-2048-F32-SINK`).** `f32` 2048 read 1.18 to 1.25 of
  RustFFT on the performance core as four 512-blocks under the radix-4
  sink while `f64` 2048 read at parity; RustFFT plans 2048 as its
  256-point butterfly under one 8xn column pass. The 256 base takes
  eight blocks under a radix-8 sink whose seven twiddle rows
  `W_2048^{j k}` come from the stage-major table (RustFFT's table
  shape; the 4096 attempt derived six powers a chunk on the sink's
  chain), gathered at eight lanes by an eight-way pair deinterleave (the
  four-way of each half of eight chunks, then one pair deinterleave
  across the halves; 43 instructions per eight registers against the
  four-way's 19 per four) and read at stride eight from the parent at
  four; the plan selects the form by width, the 512 base keeping 2048
  at four lanes. Pinned probe, two runs
  (`output/apollo-base128/small_sizes_eight2048_run{1,2}_2026-09-11.txt`),
  apollo / RustFFT: `f32` 2048 on the performance core 1.05 / 1.12
  (1482, 1495 ns against 1419, 1335; the four-block form 1637 / 1637
  against 1309 / 1381, 1.19 to 1.25), the efficiency core 1.03 / 1.03
  (2867, 2873 against 2792, 2794; the four-block form 2806 to 2943,
  inside its band); `f64` 2048 on the four-block form 2945 / 2953
  inside its 2836 to 3078 band, no other length moved. Kept: 9.5%
  under the four-block form on the performance core. Census
  (`output/apollo-base128/base256_2026-09-11.md`): the whole `f32`
  route about 21.9K instructions (the gather 2.8K, seven blocks 11.2K,
  the last block 8.0K — its column-and-sink group 874 against the
  four-block form's 938) against the four-block form's 22.6K and
  RustFFT's 26K to 28K (eight 256 butterflies of about 2.2K, the 8xn
  column loop 103 per eight registers, the transpose 31), so the route
  spends fewer instructions than RustFFT and more time: the overhead
  past eight standalone 256-blocks is 690 ns against RustFFT's 420 to
  530 past its eight butterflies. The reading: the sink phase streams
  seven spectra, seven twiddle rows, the staging buffer and eight output
  slices, twenty-three streams at 2 KB stride of which sixteen alias 4
  KB apart into one set group of the 48 KB twelve-way L1, where
  RustFFT's column pass runs eight in-place streams and one twiddle
  stream — and 1024, the same sink code over four 256-blocks with ten
  streams, pays 0.14 ns a complex of overhead against 2048's 0.34.
  Closed as a reading against its parity acceptance; the column-first
  form (RustFFT's: the radix-8 pass over the parent in place ahead of
  the blocks, the blocks out of place, one interleave pass) is filed as
  `APOLLO-2048-COLUMN-FIRST`.

- **2026-09-11, the eight blocks under a radix-8 pass ahead of them
  (`APOLLO-2048-COLUMN-FIRST`).** The stream reading held. The eight-block
  route takes RustFFT's column-first shape at either width: the radix-8
  pass over the parent in place (eight registers `BASE` samples apart,
  the register radix-8, the twiddle `W_2048^{q c}` after the butterfly
  from one chunk-major stream — the table relaid so each chunk's seven
  twiddle registers are consecutive), each eighth transformed out of
  place into scratch as a contiguous 256-block whose spectrum is
  `X[8 k + q]`, and one pass of pair interleaves back into the parent
  (two levels transposing each four-block tile at four complexes a
  register, one level pairing even and odd blocks at two). Nine streams
  a pass. The radix-8 sink, its eight-way gather and its row-major table
  are deleted. Census (`f32`, per eight registers): the radix-8 pass 97
  instructions (15 loads, 26 shuffles, 9 fused; estimated chain 39
  cycles on an issue bound of 24) against RustFFT's column loop of 103
  (34 / 24); the interleave 51 (32 shuffles) against RustFFT's transpose
  of 62. Pinned probe, two runs
  (`output/apollo-base128/small_sizes_columnfirst_run{1,2}_2026-09-11.txt`),
  apollo / RustFFT: `f32` 2048 on the performance core 0.91 / 0.96
  (1296, 1294 ns against 1423, 1354; the sink form 1482 / 1495, the
  four 512-blocks 1637 / 1637), the efficiency core 1.02 / 1.02 (2880,
  2884 against 2815, 2820; the sink form 2867 / 2873); the overhead past
  eight standalone 256-blocks 518 ns from 690, RustFFT's own. `f64`
  2048 on its four-block form 2937 / 2894 inside its band, 1024 and 4096
  inside theirs. Kept, and the item closed on its acceptance: `f32`
  2048 at or below RustFFT on both runs. What remains at 2048 is the
  efficiency core (1.02 at `f32`, 1.11 at `f64` on the four-block form)
  and at `f64` 4096 the four-step route (1.14 to 1.22), each its own
  item.

- **2026-09-11, 4096 as eight 512-blocks under the radix-8 pass ahead of
  them (`APOLLO-4096-COLUMN-FIRST`).** RustFFT plans 4096 as its 512
  butterfly under one 8xn pass (`avx_planner`: `power2 % 3 == 0` takes
  `butterfly(512)`); the column-first route is generic over the base, so
  the plan builds the 512 state at 4096 and routes `log2 = 12` through
  it at either width, the four-step boundary moving to 8192. The
  eight-512-block sink form was rejected earlier today on its working
  set and derived twiddles; the column-first form streams nine a pass
  and reads its chunk-major table. Pinned probe, two runs on a quiet
  host (`output/apollo-base128/small_sizes_eight4096_run{1,2}_2026-09-11.txt`),
  apollo / RustFFT: `f64` 4096 on the performance core 0.74 / 0.78
  (6233, 6407 ns against 8407, 8271; the four-step route 7962 to 8204),
  `f32` 0.90 / 0.87 (3045, 3077 against 3374, 3557; the four-step 3956
  to 4091); the efficiency core `f64` 0.94 / 0.93 (11663, 11444 against
  12350, 12354; the four-step 12610 to 12775) and `f32` 0.96 / 0.97
  (6174, 6178 against 6413, 6373; the four-step 7206 to 7257). 2048,
  1024 and 32768 inside their bands. Kept: 20 to 24% under the four-step
  route on the performance core and 8 to 14% on the efficiency core, at
  both scalars, and under RustFFT on every row — the first length above
  the base route to read under it at `f64` on the efficiency core. The
  ladder's next step, 8192 and above, is the four-step route (`f32`
  32768 1.06 to 1.10, `f64` 1.07 to 1.15 on the performance core), its
  own item.
  Replication of the contributor's paired instrument for the 2048 form
  (`output/apollo-base128/quiet-host/replication.md`): the
  performance-core `f32` 2048 gain over the sink reads 10.4 to 14.6%
  with disjoint intervals in six of eight comparisons across both
  experiments; the efficiency-core target (0.8 to 2.2% slower) and every
  control sit inside the band identical code shows on this host under
  concurrent compilation (1 to 4.5% on the efficiency core, 40% under
  one load spike).
