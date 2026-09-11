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
