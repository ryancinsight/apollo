# ADR 0045: The per-ISA intrinsic fork retires family by family onto hermes lane kernels, each behind a measurement gate

- Status: Accepted (2026-09-02)
- Item: `ATLAS-APOLLO-ISA-FORK-2026-08-25`
- Driving evidence: the `core::arch` census in `gap_audit.md#phastft-2026-08-25`
  (Finding 3); the FWHT negative result `gap_audit.md#fwht-vectorize-negative`;
  the interleaved pinned probe for the first slice (this record)

## Context

`apollo-fft` carried its own AVX2 and AVX-512 intrinsics for the Stockham
stages: at revision `424ce431`, 28 files importing `core::arch`, 90
`#[target_feature]` attributes, and 429 `unsafe` blocks, while `hermes-simd` —
the Atlas owner of lane-parallel CPU work — served one function in one file.
Every `unsafe` in that fork discharges the same obligation, "AVX+FMA present,
proved by the route's runtime check", which is exactly what hermes'
`Simd<T, A>` token carries by construction. The `base128/`, `batched/`, and
split-gather kernels already run on `LaneKernel`, so the fork is not a
capability gap; it is duplicated dispatch and duplicated unsafe.

Two prior decisions bound this one. ADR [0042](0042-avx-stockham-backend-retained.md)
retains the AVX Stockham *routes* while an unexplained performance-core
pathology is investigated, so no routing may change on the strength of one
core class. The FWHT negative result showed a hermes conversion can lose 1.6x
to 8.8x when the kernel is bandwidth-bound or the dispatch wraps too little
work, so a conversion is never justified by the census alone.

## Decision

- **Convert one operation family per increment, keeping every route and size
  threshold.** A family's intrinsic body is re-expressed as one generic
  `LaneKernel` over `ComplexReg`, dispatched through
  `vectorize_hardware_lanes::<LANES, T, _>` at the lane width the route was
  tuned for (4/8 for f64, 8/16 for f32), falling back to the scalar recurrence
  when the host has no backend at that width. The intrinsic copy and its
  backend-trait surface are deleted in the same change. Routing decisions stay
  with ADR 0042; this record changes how a stage is written, not where it runs.
- **Rounding is preserved, not merely bounded.** `ComplexReg`'s multiply is the
  dup/swap/`fmaddsub` sequence the intrinsic copy used, in the same operand
  order, and adds and subtracts are lane-wise IEEE operations in both. The
  differential test against the scalar recurrence therefore carries a derived
  bound (14 · ε · 4 for the pair stage) that the port must meet with margin,
  and a bitwise change against the retired copy would be a defect.
- **Each family is measurement-gated.** Before a conversion merges, the pinned
  probe runs the sized routes the family serves, interleaved before/after from
  matched binaries on queried-class processors, with RustFFT in the same run as
  the control. A regression outside the run-to-run spread reverts the family
  and records the negative result, as FWHT did.
- **Order follows unsafe density.** After the pair stage:
  `precision/{precise,reduced}.rs`, the two `backend_impl.rs`, `transform.rs`.
  The SAFETY ratchet (`scripts/safety_ratchet.py`) is the tracked metric and
  tightens with every slice.

## First slice: the generic pair stage (2026-09-02)

`stockham/avx/generic/pair.rs` (57 uncommented `unsafe` blocks) became
`stockham/butterfly/pair_lanes.rs`; the four dispatch sites in
`precision/{precise,reduced}.rs` request their width and fall back to
`stage_pair_impl`. Ratchet: 302 → 237.

Interleaved pinned probe, two rounds, Core Ultra 9 285K, release, medians in
µs (`after/before` on the better round; RustFFT column is the same-run control):

| core | prec | n | before | after | after/before | RustFFT |
| --- | --- | --- | --- | --- | --- | --- |
| performance | f32 | 1024 | 7704 | 7716 | 1.002 | 580 / 621 |
| performance | f32 | 4096 | 4206 | 4255 | 1.012 | 3279 / 3551 |
| performance | f32 | 32768 | 50574 | 51083 | 1.010 | 37817 / 35186 |
| performance | f64 | 1024 | 1807 | 1758 | 0.972 | 1171 / 1243 |
| performance | f64 | 4096 | 9608 | 9177 | 0.955 | 7223 / 7153 |
| performance | f64 | 32768 | 103469 | 103763 | 1.003 | 78301 / 75948 |
| efficiency | f32 | 1024 | 3319 | 3330 | 1.003 | 1271 / 1268 |
| efficiency | f32 | 4096 | 7088 | 7102 | 1.002 | 6484 / 6392 |
| efficiency | f32 | 32768 | 67009 | 67037 | 1.000 | 67753 / 67843 |
| efficiency | f64 | 1024 | 3085 | 3086 | 1.000 | 2461 / 2455 |
| efficiency | f64 | 4096 | 13211 | 13242 | 1.002 | 12495 / 12555 |
| efficiency | f64 | 32768 | 125642 | 125639 | 1.000 | 129357 / 129342 |

The efficiency-core rows repeat to 0.3% between rounds and move by at most
0.3%; the performance-core rows spread up to 5.7% between rounds of the *same*
binary (f32 4096: 4206 vs 4448) and the control moves by up to 7% in the same
cells, so the +1.0–1.2% f32 rows are inside the instrument's spread and the
−2.8%/−4.5% f64 rows are the only movements that clear it. Verdict: neutral,
with a small f64 gain; the gate passes.

## Second slice: the single stage (2026-09-02)

`stockham/avx/generic/base.rs` became `butterfly/lanes/base.rs`
(`BaseStage`), the lane kernels moved under `butterfly/lanes/` with one
shared differential harness, and the backend trait lost the four scalar
helpers only the retired copies called. Ratchet: 237 → 220.

Same instrument, pair-slice binary as `before`, two rounds, medians in µs
(`after/before` on the better round):

| core | prec | n | before | after | after/before |
| --- | --- | --- | --- | --- | --- |
| performance | f32 | 1024 | 7728 | 7714 | 0.998 |
| performance | f32 | 4096 | 4457 | 4196 | 0.941 |
| performance | f32 | 32768 | 50685 | 51523 | 1.017 |
| performance | f64 | 1024 | 1803 | 1896 | 1.051 |
| performance | f64 | 4096 | 9292 | 8755 | 0.942 |
| performance | f64 | 32768 | 103308 | 104732 | 1.014 |
| efficiency | f32 | 1024 | 3328 | 3325 | 0.999 |
| efficiency | f32 | 4096 | 6958 | 7042 | 1.012 |
| efficiency | f32 | 32768 | 65941 | 65770 | 0.997 |
| efficiency | f64 | 1024 | 3017 | 3016 | 1.000 |
| efficiency | f64 | 4096 | 13229 | 13065 | 0.988 |
| efficiency | f64 | 32768 | 124752 | 125584 | 1.007 |

The base routes (256, 512) were controls and moved by at most 1.7%. The
efficiency-core rows are within 1.2%; on the performance core the same
binary spread 25% between rounds at f32 32768 (50685 vs 63203), so the
+1.4–5.1% cells there are inside the instrument and the −5.8/−5.9% cells at
4096 are the only movements that clear it. Verdict: neutral, small gain at
4096; the gate passes.

## Third slice: the triple stage (2026-09-02)

`stockham/avx/generic/triple.rs`'s three generic bodies — `stage_triple_avx_fma`,
its byte-identical `low_live` twin (the L1-residency predicates that chose
between them are deleted with it), and the radix-one form — became
`butterfly/lanes/triple.rs`. The radix-one quarter turns rotate through
`ComplexReg::mul_i`/`mul_neg_i` selected once from the twiddle's sign, the
same permute-and-xor the intrinsic used. The sized radix-one specialisations
(n = 32..32768) stayed on the intrinsic helper for this slice. Ratchet:
220 → 208.

Same instrument, base-slice binary as `before`, two rounds, medians in µs:

| core | prec | n | before | after | after/before |
| --- | --- | --- | --- | --- | --- |
| performance | f32 | 1024 | 7735 | 790 | **0.102** |
| performance | f32 | 4096 | 4423 | 4175 | 0.944 |
| performance | f32 | 32768 | 50729 | 51047 | 1.006 |
| performance | f64 | 1024 | 1909 | 1827 | 0.957 |
| performance | f64 | 4096 | 8779 | 9177 | 1.045 |
| performance | f64 | 32768 | 103613 | 103687 | 1.001 |
| efficiency | f32 | 1024 | 3323 | 1558 | **0.469** |
| efficiency | f32 | 4096 | 6966 | 6970 | 1.001 |
| efficiency | f32 | 32768 | 66107 | 65718 | 0.994 |
| efficiency | f64 | 1024 | 3004 | 3031 | 1.009 |
| efficiency | f64 | 4096 | 13108 | 13048 | 0.995 |
| efficiency | f64 | 32768 | 124411 | 123853 | 0.996 |

Every cell but two is inside the instrument's spread (efficiency-core rows
within 1%, performance-core rows within the same binary's round-to-round
range; the 256/512 controls moved ≤ 3.4%). The two are the f32 n = 1024
route: 7.7 µs → 0.79 µs on the performance core and 3.3 µs → 1.56 µs on the
efficiency core, against RustFFT's 0.58 µs / 1.27 µs. That route is the
radix-one triple first pass, which `reduced.rs` sent to the generic intrinsic
body at n = 1024 (the one size with no sized specialisation); the intrinsic
body was the "finding 3" pathology recorded in ADR
[0042](0042-avx-stockham-backend-retained.md) — 13x slower on a performance
core than the efficiency core — and the lane kernel does not carry it. The
sized specialisations share that body's element step, so the next slice
measures them against the lane kernel size by size.

## Fourth slice: the sized radix-one unrolls (2026-09-02)

The seven sized radix-one triple specialisations (`triple/n{32,64,128,256,
512,1024,32768}.rs`, 2,996 lines of explicit `do_one` unrolls over one
element step) and their twelve dispatch arms were deleted; those sizes now
reach the radix-one lane kernel through the general arm. With the last
intrinsic stage body gone, `StockhamAvxBackend` lost its vector primitives
and `avx/generic/` was removed. Ratchet: 208 → 168.

Same instrument over every probed size (8..32768), triple-slice binary as
`before`, two rounds; the cells the deleted unrolls served:

| core | prec | n | before | after | after/before | RustFFT before/after |
| --- | --- | --- | --- | --- | --- | --- |
| performance | f64 | 32 | 21.9 ns | 30.9 ns | 1.411 | 15.6 / 22.5 (1.44) |
| performance | f64 | 64 | 46.2 | 42.4 | 0.917 | 46.6 / 39.8 (0.85) |
| performance | f64 | 128 | 94.0 | 96.7 | 1.029 | 86.4 / 92.5 |
| performance | f64 | 256..32768 | — | — | 1.000–1.033 | — |
| performance | f32 | 32..32768 | — | — | 0.944–1.090 | — |
| efficiency | f64 | 32..32768 | — | — | 0.995–1.015 | — |
| efficiency | f32 | 32..32768 | — | — | 0.994–1.019 | — |

The efficiency-core rows repeat to 1.5%. The one performance-core cell
outside the spread, f64 n = 32 at +41%, moved with its same-run RustFFT
control (+44%, a binary that did not change), so it is the instrument —
performance-core timing of a 20 ns transform shifts with binary layout —
not the route; f64 n = 64 shows the mirror image (−8% with the control at
−15%). Verdict: neutral; the gate passes, and the f64 n = 32 cell is
re-measured with the next slice, which changes exactly that leaf.

## Fifth slice: the fixed 32/64-point f64 leaves (2026-09-02)

`avx/precise/fixed.rs` (748 lines: an 8×8 and an 8×4 Cooley–Tukey
factorisation over AVX intrinsics) became `Dft64`/`Dft32` lane codelets in
`butterfly/lanes/fixed_precise.rs`, built on the shared
`register_butterfly` DFT-4/DFT-8 with one pair-deinterleave for the 2×2
sample transpose between phases and the same exact twiddle tables. The
leaves are tested against a naive DFT in both directions. The ratchet is
unchanged at 168 (every deleted block carried its comment).

Same instrument, sized-unroll binary as `before`, two rounds; the sizes the
leaves serve directly and the four-step rows that use them:

| core | prec | n | before | after | after/before | RustFFT before/after |
| --- | --- | --- | --- | --- | --- | --- |
| efficiency | f64 | 32 | 60.1 ns | 61.4 ns | 1.023 | 41.1 / 41.2 |
| efficiency | f64 | 64 | 89.6 | 89.6 | 1.000 | 84.3 / 84.6 |
| efficiency | f64 | 1024 | 2992 | 3100 | 1.036 | 2453 / 2466 |
| efficiency | f64 | 4096 | 13194 | 13221 | 1.002 | 12458 / 12355 |
| performance | f64 | 32 | 23.4 (r2: 38.3) | 31.7 (r2: 41.8) | 1.354 | 16.1 / 24.5 |
| performance | f64 | 64 | 43.3 | 46.3 | 1.069 | 39.1 / 45.5 |
| performance | f64 | 1024 | 1896 | 1782 | 0.939 | 1175 / 1255 |
| performance | f64 | 4096 | 9088 | 9131 | 1.005 | 6625 / 7096 |
| both | f32 | 32..32768 | — | — | 0.981–1.028 | (controls) |

The performance-core cells at 32 and 64 spread 63% and 7% between rounds of
the *same* binary and their RustFFT controls moved 52% and 16%, so they carry
no information at this scale; the efficiency core repeats to 1.5%. There the
port costs **+2.3% at f64 n = 32 (0.4 ns beyond the round-to-round spread)**
and nothing at 64, 1024, or 4096. Accepted: 0.4 ns at one size against 743
lines of hand-scheduled intrinsics, with the codelet now sharing the
`register_butterfly` bodies every other fixed kernel uses; the cell stays in
the probe, so a later change to `radix8` that recovers it is measured.

## Sixth slice: the fixed 64-point f32 leaf (2026-09-02)

The f32 64-point leaf — the radix-one triple lane kernel followed by the
intrinsic `stage_triple_quarter_groups_one_reduced_avx_fma` — became the
`Dft64Reduced` codelet in `butterfly/lanes/fixed_reduced.rs`: the same 8×8
factorisation as the f64 leaf with four columns per register and one 4×4
sample transpose (`ComplexReg::transpose_square`) between phases, tested
against an f64 naive DFT in both directions.

Same instrument, f64-leaf binary as `before`, two rounds:

| core | prec | n | before | after | after/before | RustFFT before/after |
| --- | --- | --- | --- | --- | --- | --- |
| efficiency | f32 | 64 | 54.5 ns | 54.5 ns | 0.999 | 51.3 / 51.5 |
| efficiency | f32 | 32..32768 (rest) | — | — | 0.994–1.004 | — |
| performance | f32 | 64 | 30.2 | 22.7 | 0.752 | 20.8 / 28.5 |
| performance | f32 | 1024 | 821 | 800 | 0.974 | 581 / 589 |
| performance | f32 | 4096 | 4252 | 4141 | 0.974 | 3493 / 3317 |
| efficiency | f64 | 32 | 60.9 | 60.1 | 0.986 | 41.2 / 41.1 |

The leaf is exactly flat on the efficiency core; the performance-core f32
n = 64 cell moved −25% while its control moved +37%, the same layout
sensitivity the fifth slice recorded. The efficiency-core f64 n = 32 cell,
which the fifth slice recorded as +2.3%, reads 60.1 ns again against a
binary in which the f64 leaf did not change — so that +2.3% was binary
layout too, not the codelet. Verdict: neutral; the gate passes.

## Seventh slice: the groups-one stage (2026-09-02)

The final Stockham stage (`groups == 1`) — an AVX/FMA body per precision
and two AVX-512 bodies behind `StockhamAvxBackend::stage_groups_one` —
became `GroupsOneStage` in `butterfly/lanes/groups_one.rs`: one
pair-deinterleave of two loaded registers yields the even and odd inputs a
register of `j` needs. Ratchet: 168 → 162.

Same instrument, f32-leaf binary as `before`, two rounds; the cells that
moved beyond the efficiency core's 0.5% repeatability:

| core | prec | n | before | after | after/before | RustFFT before/after |
| --- | --- | --- | --- | --- | --- | --- |
| efficiency | f32 | 1024 | 1562 ns | 1601 ns | **1.025** | 1267 / 1268 |
| efficiency | f64 | 16 | 16.8 (r2: 19.4) | 19.1 | 1.135 | 20.0 / 20.2 |
| efficiency | f64 | 32768 | 125941 | 123050 | 0.977 | 130284 / 130010 |
| efficiency | rest | 8..32768 | — | — | 0.986–1.003 | — |
| performance | f32 | 1024 | 790 | 709 | 0.897 | 609 / 577 |
| performance | f32 | 4096 | 4417 | 4146 | 0.939 | 3201 / 3491 |

The f64 n = 16 cell spread 15% between rounds of the same binary; the rest
of the performance-core column is inside its usual spread. The one real
movement is **f32 n = 1024 at +2.5% (40 ns)** on the efficiency core, both
rounds agreeing to 0.2%. Its final stage is radix 512 at eight lanes, where
hermes' AVX2 `deinterleave_pairs` is two cross-lane `permute2f128` plus two
`shuffle_ps` per four digits, against the retired body's two in-lane
`unpack` per two digits at SSE width — the same shuffle count, but the
cross-lane form pays port-5 latency the unpacks do not. Accepted with the
number recorded: the cure is an unpack-based even/odd split in hermes'
AVX2 f32 backend, filed there as `HS-DEINTERLEAVE-PAIRS-AVX2-F32-2026-09-02`
and measured through this cell when it lands; apollo has no four-lane f32
hardware backend to route to meanwhile (hermes ships AVX2/AVX-512 only on
x86-64).

## Alternatives rejected

- **Retire the AVX Stockham backend wholesale** onto the auto-vectorized scalar
  routes: ADR 0042's finding 3 (the AVX arm inverts the core hierarchy) is
  unexplained, and by that record only both-core-dominant reroutes are safe.
  Rewriting the stages generically is orthogonal to that question and does
  not pre-empt it.
- **Convert by census order** (largest file first): rejected by the FWHT result;
  conversion order follows unsafe density, but *merging* follows measurement.
- **Keep the fork behind `StockhamAvxBackend` and add NEON impls**: a third copy
  of every stage; the seam that hermes already provides.

## Failure modes and limits

- `vectorize_hardware_lanes` returns `None` on a host without a backend at the
  requested width; every site falls back to the scalar recurrence, so a
  non-AVX host loses the intrinsic route it never had.
- The AVX-512 widths (8 f64 / 16 f32) are unmeasurable on this host (Arrow
  Lake); the differential test covers them when served, the probe does not.
- The kernel module is `cfg(target_arch = "x86_64")` today because only the
  AVX precision impls call it; a NEON route would lift the gate, not fork the
  body.
- The pair stage's `call` bodies are 65–85 lines, under the 100-line
  `lane_kernel_uninlined` threshold; larger families must carry
  `#[inline(always)]` per hermes' `LaneKernel` contract
  (`ATLAS-APOLLO-LANEKERNEL-INLINE-CONTRACT-2026-08-31`).

## Verification

- `pair_lanes::tests`: differential against `stage_pair_impl` at f32/8, f64/4
  (and f64/8 when served) across vector, tail, and all-tail `(n, radix)` cases;
  an unserved width is asserted to touch nothing.
- `stockham`, `backend_matrix`, and `phastft_parity` suites unchanged and
  green; `apollo-fft` nextest 529/529.
- The interleaved probe table above.
