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

Recommended shape (option A below): sixteen 16-point columns as two
radix-4 passes through the staging buffer (the base-128 kernel's phase 1
and 2, widened from eight rows to sixteen), the transpose in registers on
the way into staging, then sixteen 16-point rows as the column pass
widened from eight to sixteen points. The row twiddles stay broadcast
scalars; the internal twiddle table grows from 8 x 16 to 16 x 16.

## Options

- **A. A 256 base as 16 x 16 in the base-128 kernel's shape** (recommended).
  Reuses the two-pass structure, the staging discipline, the sinks and the
  plan tables; the register budget per pass is the base-128 kernel's
  (about a dozen live values per stage). Risk: the column pass over
  sixteen points needs a 16-point DIF in registers where the kernel has
  an 8-point one; the sixteen-point step is the radix-4 pair the rows
  already use.
- **B. A 256 base as 8 x 32, RustFFT's shape.** Eight-point columns keep
  the existing column pass; the rows become 32-point (`dft32` exists as
  a scalar Winograd codelet, not as a register kernel). Risk: a 32-point
  row kernel over registers is new work with no sibling to copy, and the
  8 x 32 twiddle layer is wider than 16 x 16's.
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

None; Proposed.
