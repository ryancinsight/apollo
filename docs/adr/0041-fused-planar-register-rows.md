# ADR 0041: Fused planar-register rows for the mid-size four-step

- Status: Proposed
- Item: ATLAS-APOLLO-FUSED-PLANAR-ROWS
- Date: 2026-08-27
- Revision 2026-08-27: increment one (the planar row body, built behind the
  full oracle set and measured pinned) **falsifies the planar-register
  premise**. Planar rows measured ~6.3k TSC per pass against the interleaved
  kernel's ~5.6k: the boundary deinterleave/interleave networks plus the two
  in-register stages cost as many shuffle-class ops (~96/row) as the
  interleaved multiply chain they replace, and the P-core's shuffle
  throughput means the interleaved form was never shuffle-port-bound. Both
  register-row shapes pay ~3x a batched streaming pass in arithmetic alone.
  RustFFT's construction, read from `avx_planner.rs` at the locked version
  (the f64 impl — an earlier 256-base reading came from the f32 arm, the
  correction owed to the independent PR #147 review), is
  `MixedRadix8xnAvx(Butterfly128Avx64)` — a hand-written 128-point
  interleaved AVX base butterfly composed with an 8xn mixed-radix layer that
  reserves full-length scratch and performs an explicit transpose. The
  winning shape is therefore a **large L1-resident interleaved base
  transform with a short radix chain and honest scratch**, not per-row
  register residency and not planar streaming past its current form.
  Decision section superseded accordingly; the planar module stays test-gated
  as the differential oracle and measurement instrument. Five-engine pinned
  standing at N = 1024 P-core: batched 3.4-3.7 us, resident 6.7, planar 7.4,
  RustFFT 2.47, PhastFT 3.25.

## Context

The register-resident interleaved row experiment (`components/resident`,
PRs #141/#143) is correct and, after the hermes dispatch-inline fix
(hermes PR #78), runs its rows near their port-limited bound. Its per-pass
TSC attribution at N = 1024 pinned P-core is the governing evidence:

| pass | TSC cycles |
| --- | --- |
| rows1 (interleaved DIF) | 5 462 |
| rows2 (matrix fold + DIF) | 7 659 |
| transpose x2 | 2 799 |
| untangle involution | 5 086 |

Two structural facts follow. First, the interleaved rows alone (13.1k)
exceed the batched route's entire budget (~11k TSC = 3 423 ns): the
interleaved complex multiply costs three shuffles feeding one
fmaddsub, and the shuffle ports saturate long before the FMA ports —
~240 shuffle-class ops per 32-point row. Second, the shape spends 38%
of its time moving data between row passes (two transposes plus the
closing bit-reversal involution, 7.9k TSC). RustFFT's measured
2 455 ns (~7.9k TSC) on the same probe is the existence proof that the
four-step at this size can run in about the cost of our two row passes
alone — it pays neither tax.

## Decision

Build the mid-size power-of-two kernel as **planar-register rows with
transposes fused into the row load/store networks**:

1. **Planar registers, same sixteen-register fit.** A 32-sample row is
   held as 8 real + 8 imaginary ymm registers instead of 16 interleaved
   ones. Every butterfly and twiddle multiply becomes pure add/mul/FMA
   traffic — zero shuffles in all five stages — moving the row body from
   the saturated shuffle ports to the FMA ports (2/cycle). Stage
   structure, DIF ordering, and the rev-baked four-step matrix carry
   over from the resident design unchanged.
2. **Fused boundary networks.** The array transposes and the
   interleaved-to-planar conversion are one shuffle network executed at
   row load, and its inverse at row store; the closing bit-reversal
   rides the store addressing exactly as the matrix fold rides the load.
   The transform touches the array twice (two row passes) instead of
   five times. The unpack networks reuse the `ComplexReg`
   interleave/deinterleave and `swap_pairs` vocabulary hermes already
   ships; missing permute primitives are implemented upstream in hermes
   per upstream ownership, never approximated downstream.
3. **Routing unchanged until the bar is met.** The kernel dispatches
   behind the existing pot routing with fallback to batched for any
   unhandled width or length. It takes over the route only at or below
   RustFFT's probe number; at or below batched it may merge test-gated
   as the resident module does today.

## Alternatives

- **Keep tuning the interleaved resident shape.** Rejected: the shuffle
  tax is representational. No scheduling of a 3-shuffle multiply
  reaches the 2/cycle FMA ceiling the planar form starts at.
- **Keep tuning the batched planar streaming route.** Retained as the
  production incumbent and fallback, but its pass count (five streamed
  stage passes plus deinterleave) bounds it near its current ~11k TSC;
  the references win by touching memory twice.
- **AVX-512 lanes.** Not available on the measurement host (Arrow Lake
  has no AVX-512); the design monomorphizes over lane count through the
  hermes seams, so wider ISAs inherit the construction when present.

## Verification plan

The pinned four-engine same-process probe (`resident/pinned_probe.rs`)
is the acceptance instrument, N = 1024, P-core and E-core:
merge-as-experiment at <= batched (3 423 ns P); route takeover at
<= RustFFT (2 455 ns P). Correctness inherits the resident oracle set:
direct-DFT both directions, round trip, batched differential,
decline-untouched. Numerical bounds per the existing derived tolerances
(reduction-order-sensitive, epsilon-bounded).
