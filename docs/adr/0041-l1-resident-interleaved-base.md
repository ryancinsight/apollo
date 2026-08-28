# ADR 0041: L1-resident interleaved base transforms

- Status: Accepted
- Items: ATLAS-APOLLO-BASE-BUTTERFLY-128,
  ATLAS-APOLLO-FUSED-PLANAR-ROWS
- Date: 2026-08-27
- Revision 2026-08-27: rewrites the rejected planar-row proposal around the
  measured interleaved 128-point base. The prior record retained a superseded
  Proposed decision beside contrary evidence. This revision makes the current
  decision authoritative and separates admissible comparison timing from
  diagnostic phase instrumentation.
- Revision 2026-08-27: records rejection of a premature production route that
  kept the experiment's thread-local plan instead of moving ownership into
  `FftPlan1D`. The unchanged exact comparator found repeatable unrelated-case
  regressions, so production remains on the incumbent route.

## Context

Apollo's planar batched power-of-two route is competitive once its streamed
passes amortize, but fixed deinterleave, transpose, permutation, and
reinterleave traffic dominates small and mid-size transforms. At N = 1024,
the arithmetic stages already fit below RustFFT's whole-transform cycle budget;
about 45% of Apollo's measured budget was movement.

Two register-row experiments tested whether two fused 32-point row passes could
remove that traffic. Both were slower than the batched route. The interleaved
shape saturated shuffle throughput; the planar shape replaced multiply
shuffles with equally costly boundary shuffle networks. Those measurements
reject per-row register residency as the production architecture.

The locked RustFFT f64 planner instead composes a hand-written interleaved
128-point AVX base with an 8-by-N mixed-radix layer and full-length scratch.
That source and Apollo's pass attribution identify the missing mechanism: a
larger L1-resident base shortens the radix chain and accepts one explicit
redistribution instead of repeatedly streaming the full transform.

## Decision

1. **Use one interleaved mixed-radix 8-by-16 base.** The first pass
   redistributes eight stride-8 subsequences into a 2 KiB staging array. Its
   block-pair stores absorb the DIT-16 bit reversal. Eight register-resident
   DIT-16 rows then feed a twiddled lane-wise DIF-8 column pass whose stores are
   contiguous and naturally ordered. Dup-split twiddles reduce each general
   interleaved complex multiply to one shuffle, one multiply, and one
   alternating FMA.
2. **Treat lane width as a capability, not an assumption.** The current map
   requires four scalar lanes, or two interleaved complex samples. Hermes
   `vectorize_lanes::<4, T, _>` selects f64 AVX2 even on AVX-512 hosts and
   selects f32 NEON or the portable packed backend. A host without an exact
   match declines before base/resident plan construction or mutation.
3. **Keep plans immutable and outside the hot call.** The test-gated experiment
   initializes each direction's 496-lane twiddle table once in a thread-local
   `OnceCell`, stores it as `Box<[T]>`, and borrows it for each call. This
   removes the previous per-transform `Arc` atomic increment and extra plan
   allocation. Production integration moves the same immutable plan into the
   owning FFT plan; thread-local ownership is an experiment boundary, not the
   final route contract.
4. **Measure the comparison without attribution code.** The pinned instrument
   uses Apollo Bench's 100 ordered samples and exact distribution-free median
   interval. Its timed base specialization contains no TSC stamps or atomic
   counters. A distinct const specialization performs serialized TSC phase
   attribution after the comparison; compile-time elimination, followed by
   codegen inspection, is the zero-instrumentation acceptance oracle.
5. **Route only on complete evidence.** The base may remain test-gated when its
   median interval clears the batched route. It replaces the production N = 128
   route only after forward, inverse, round-trip, incumbent differential,
   supported-width, and clean-decline contracts pass and its interval clears
   that route. The 8-by-128 N = 1024 composition begins only after the corrected
   base measurement makes its outer redistribution budget viable.

## Failure modes and controls

- A width mismatch could silently run the address map with the wrong number of
  complex samples per register. Exact-count dispatch filters before invoking
  the kernel, the kernel retains its local `LANE_COUNT == 4` invariant, and
  tests assert execution or bit-preserving decline.
- A benchmark could compare instrumented Apollo code with uninstrumented
  references. Const-specialized timing and attribution paths plus codegen
  inspection prevent that recurrence.
- Bounds checks or host-support probes could re-enter the hot loop. Capability
  views and token-scoped zero/splat constructors provide all constants and
  loads after one dispatch.
- A per-call plan handle could add atomic and allocator traffic at N = 128.
  Immutable cached borrowing removes both from the experiment; plan ownership
  is explicit in the production integration requirement.
- The direct DFT oracle and FFT may differ by expected rounding order. Bounds
  use `gamma_k = ku / (1 - ku)` with operation counts for both computations;
  bitwise equality is not claimed.

## Alternatives

- **Fused planar-register rows.** Rejected by the measured boundary-shuffle and
  row-arithmetic cost; the former Proposed decision is superseded.
- **Continue only with streamed batched tuning.** Retained as the incumbent and
  fallback, but rejected as the sole architecture because its movement floor is
  the measured small-size bottleneck.
- **Copy RustFFT's AVX implementation.** Rejected. Apollo keeps one Hermes
  kernel and implements missing first-party SIMD capabilities upstream rather
  than binding algorithm code to vendor intrinsics.
- **Route the four-lane experiment after dispatch alone.** Rejected until
  production plan ownership, statistical timing, and every value oracle close
  together.

## Evidence and limits

The direct DFT establishes value behavior; round trip and incumbent
differential add independent transform and integration checks. Apollo Bench
establishes pinned same-process timing with exact median intervals. Serialized
TSC counters locate phase cost but do not establish wall-clock speedup. Codegen
inspection establishes absence of probes and attribution instructions, not
runtime latency. Hermes PR #86 establishes the exact-width provider contract;
Apollo's hosted AVX-512 execution remains the integration gate for this
consumer revision.

The corrected 100-sample run completed its measurement body in 11.98 seconds
and produced 96.4799% exact distribution-free median intervals (nanoseconds):

| core | incumbent Apollo | base | RustFFT | PhastFT |
| --- | ---: | ---: | ---: | ---: |
| P | 687.152 [686.937, 687.564] | 294.518 [294.275, 294.826] | 181.788 [181.591, 181.986] | 330.019 [329.688, 330.503] |
| E | 1844.331 [1843.457, 1845.866] | 146.401 [146.364, 146.468] | 84.694 [84.670, 84.726] | 148.974 [148.899, 149.065] |

Thus the base is 2.33x and 12.60x faster than the incumbent route on the
measured P- and E-cores. It is 10.8% and 1.8% faster than PhastFT, with
disjoint intervals, while RustFFT remains 1.62x and 1.73x faster. Separately,
serialized phase attribution reported 164/521/452 TSC on the P-core and
96/232/300 TSC on the E-core for redistribution/row/column work. Those phase
counts are diagnostic only.

The comparison-specialization regression test leaves all attribution counters
at zero. Source inspection finds one `LFENCE`/`RDTSC` stamp implementation and
four calls guarded by the measurement const parameter; optimized-binary
inspection finds exactly four `LFENCE`/`RDTSC` pairs. The Windows test binary
is stripped, so the disassembly cannot provide symbol-level attribution; the
const-specialized source, counter test, and instruction count jointly support
the zero-instrumentation claim.

A later direct-routing candidate did not satisfy this decision. Hosted run
33122650730 found regressions in all four paired comparisons for generic-prime
N = 31 (1.81-3.85%) and compact N = 96 (2.22-6.02%). Normalized instruction
streams for the measured hot functions were unchanged while candidate `.text`
grew by 400 bytes. This evidence is consistent with placement sensitivity, not
a changed kernel, and does not override the comparator. The candidate also
retained the experiment's thread-local plan, contrary to Decision 3. The route
was therefore withdrawn while the base and its value oracles remained
test-gated.

## Revision history

- 2026-08-27: Reject the premature thread-local production route after the
  exact benchmark comparator found repeatable unrelated-case regressions;
  retain plan ownership and comparator clearance as production requirements.
- 2026-08-27: Record merged Hermes PR #86 exact-count dispatch and remove the
  resolved provider-width blocker. Independent review moved resident
  capability resolution ahead of plan construction and input permutation;
  production plan ownership remains open.
