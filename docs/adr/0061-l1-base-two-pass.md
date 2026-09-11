# ADR 0061: A two-pass base of 256 for the L1-resident powers of two

- **Status:** Proposed
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
measured first and gained nothing over the split at 256.

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
