# 0062 — Composite bases as register kernels under column passes

- Status: Accepted
- Date: 2026-09-11
- Item: `backlog.md#apollo-codelets-over-lanes` (parent
  `backlog.md#atlas-apollo-beat-the-references`)
- Evidence: `output/apollo-base128/composite_census_2026-09-11.md`,
  `output/apollo-base128/small_sizes_kernelgap2_run{1,2}_2026-09-11.txt`
  (before), `output/apollo-base128/small_sizes_column180_run{3,4}_2026-09-11.txt`
  (the route), `output/apollo-base128/base256_2026-09-11.md` (the census)

## Context

With the small-length call path closed and hermes' sign-flip rotation in
every register kernel, 180 is the widest gap left on the scoreboard: 1.43 to
1.47 of RustFFT at both scalars on the performance core, 1.25 to 1.27 on the
efficiency core. The composite census read the reason: apollo's route runs
four flat Stockham passes (`[4, 3, 3, 5]`), each a full read and write of the
array and a twiddle load per register, and its per-register butterflies cost
what RustFFT's do (R3 30 against 34 instructions, R4 39 against 44, R5 56 to
64 against 63). RustFFT runs two stages: `Butterfly36Avx`, a 36-point
transform held in registers, under one `MixedRadix5xnAvx` column pass.

The spike asked whether the generated Winograd codelets (`dft36_impl` and
its family) could be instantiated over lane vectors — one codelet call
transforming four or eight columns at once — and serve as that base.

## Findings

**The codelet bound does not admit a lane type.** The generator
(`apollo-fft-macros::generate_winograd_composites!`) emits
`F: WinogradScalar + ShortDft<n1> + ShortDft<n2>`. `WinogradScalar` is
`RealField + PartialOrd + PrimePairTables + …` — a real-field contract with
ordering, `NAN`, `INFINITY`, `abs`, and the per-scalar prime tables — and
`ShortDft<N>: ShortWinogradScalar` carries eighteen per-type methods, several
of them hand-written SSE bodies per scalar (`dft3` for `f32`). A vector of
lanes can implement the arithmetic (`eunomia::Complex<T>` bounds its
operators on the std traits alone) and `from_precise` (a splat), but not the
field contract, and its prime tables would have to be built at run time from
the scalar tables since a vector cannot be a `const`. Admitting lanes is a
restructure of the codelet trait hierarchy: an arithmetic-and-constants root
under `ShortWinogradScalar`, the prime tables lifted to scalar tables splat
at use, and the per-type prime bodies made generic — an `[arch]` change to
every codelet consumer for a base the next finding disfavours.

**The lane dimension at 180 is five.** With `n = 36 × 5` the columns a
lane-parallel 36-point transform would run at once number `n2 = 5`: five of
eight `f32` lanes, five of eight `f64` lanes across two registers. The
generated body also stages its Good-Thomas gather and scatter through a
scratch of `n` complexes with modular index arithmetic; over lanes that is
36 register stores and loads each way per transform. The lane form pays for
three idle lanes in eight and a scratch round trip, before the transposing
gather that turns rows of five into lane vectors.

**RustFFT's base is the transform within registers, not across them.**
`Butterfly36Avx` loads four rows of nine as one partial and two full
registers each, runs the radix-4 lanewise across the four row registers,
applies `W_36`, transposes the 9×4 tile to 4×9 in registers, and runs the
radix-9 (two radix-3 stages with `W_9`) lanewise across nine registers —
every lane carrying a sample of the same transform. `MixedRadix5xnAvx`
transposes the parent's rows of five into five contiguous 36-blocks in
registers (`transpose5_packed`), runs the base on each, and applies one
twiddled radix-5 column pass across the five spectra, storing natural order.
That is the block-and-sink shape of ADR 0061 at a composite length, with a
5-way register transpose where the power-of-two routes read the parent at a
stride.

## Decision

Reject the generated codelets over lane vectors. Take the register-kernel
form for the composite bases: hand-written 2·3·5-smooth kernels over hermes
`ComplexReg` (as the DFT-16 and DFT-32 kernels in
`winograd/composite/radix_four_eight.rs`), beginning with 36, and a column
route `n = base × r` that transposes the parent into `r` contiguous blocks in
registers, runs the base kernel per block, and applies the twiddled radix-`r`
column pass writing natural order. The route lives beside the split routes
of ADR 0061 and is selected per length where it measures under the composite
route on the pinned probe.

First instance: 180 as 36 × 5 at `f32` (four complexes a register). The
`f64` kernel (two complexes a register, a row of nine as four and a half
registers) is its own layout and follows as the second increment.

## Measured route

The 180 route at `f32` (`kernel/components/column_route`, the 36-point
kernel in `winograd/composite/radix_four_nine.rs`), on hermes' five-way
pair interleave and in-frame lane-kernel entry (hermes PR #174). Pinned
probe, two runs on the rebased tree, apollo / RustFFT:

| core | before (composite) | the route |
| --- | --- | --- |
| performance `f32` 180 | 1.47 / 1.43 (131 us) | 0.94 / 1.03 (94.1, 93.5 us against 100.4, 90.7) |
| efficiency `f32` 180 | 1.25 / 1.25 (256 us) | 0.98 / 0.98 (199.8, 199.9 us against 204.8, 204.6) |

Every other length holds its reading. Census of the route (`chain.py`):
499 instructions in one frame — the 36-point body 189 against RustFFT's
`Butterfly36Avx` body of 178 (apollo 22 multiplies and 10 fused
multiply-adds against 10 and 28, thirteen stack moves in the body), the
column pass 51 against RustFFT's 5xn body of 63. Kept: 28% under the
composite route on the performance core and at RustFFT on the efficiency
core, the residue the kernel's fused-multiply count.

- **2026-09-11, the layout at two complexes a register
  (`APOLLO-COLUMN-ROUTE-F64-180`).** The 36-point kernel gains the six-by-six
  form (a row of six as three registers of two, a radix-6 as a three-by-two
  Good-Thomas across the six row registers, nine 2x2 transposes, the
  radix-6 across the six column registers — RustFFT's `Butterfly36Avx64`
  shape) selected by the register's complex count, and the column pass and
  interleave run at either width from tables built for the frame's lane
  count; the plan builds the route wherever `LaneScalar::FRAME_LANES` is
  four or eight. Pinned probe, two runs
  (`output/apollo-base128/small_sizes_column180f64_run{1,2}_2026-09-11.txt`),
  apollo / RustFFT: `f64` 180 on the performance core 1.06 / 0.68 (155.6,
  148.8 us against RustFFT's 146.5 / 218.7, the composite route's 222) and
  0.95 / 0.94 on the efficiency core (329.7, 329.1 against 348.2 / 348.5,
  the composite route's 443). Kept. Census: the `f64` route 339
  instructions in one frame, the kernel body 178 with 52 stack moves
  against RustFFT's `Butterfly36Avx64`; the `f32` route unchanged. The
  `f32` 32 efficiency-core row moved 18.1 to 20.9 us between the run pairs
  with its body's census identical (133 vector instructions, every class
  the same): code placement, not the kernel.

## Alternatives

- Codelets over lane vectors: rejected above on the bound, the lane
  utilization at 180, and the scratch traffic of the generated body.
- A radix-36 flat pass: the same kernel run over strided columns inside the
  composite core, no transposes. The gather at stride five is the cost the
  register transpose removes; kept as the fallback reading if the transposed
  route loses.
- The status quo: four passes at 1.43 to 1.47 of RustFFT.

## Verification

- The 36-point kernel against `dft36_impl` in both directions and at an
  impulse in every position, within the bound the twiddle rounding derives.
- The 180 route against the composite route and the direct DFT oracle.
- The pinned probe (`small_sizes_against_the_references_by_core_type`), two
  runs, both scalars, both cores; kept only under the composite route at
  every reading.
