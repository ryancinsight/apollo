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
