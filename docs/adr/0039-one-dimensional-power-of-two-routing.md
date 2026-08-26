# ADR 0039: One-dimensional power-of-two routing

- **Status:** Accepted
- **Date:** 2026-08-26
- **Class:** [patch] [arch]
- **Item:** `ATLAS-APOLLO-BATCHED-1D-UNREACHABLE-2026-08-26`

## Context

At the entry revision, Apollo's one-dimensional power-of-two plans called
`MixedRadixScalar::pot_inplace` without entering the generic mixed-radix
dispatcher that owned the four-step threshold. Consequently every 1-D length
above the small codelets remained on Stockham, while the batched four-step
kernel was reachable only from 2-D and 3-D lane transforms. The decision below
removes that discrepancy.

Runtime instrumentation established that boundary: no 1-D transform from 8
through 262144 entered `four_step_fft`, while a 2-D transform with a 4096-long
axis entered it immediately. The four-engine census therefore measured the
Stockham route, not the new batched route. At the same revision, warm complex
execution allocated zero bytes per call, so allocation is not the cause of the
1-D complex throughput gap.

The existing Stockham audit also excludes dispatch, twiddle lookup, and scratch
acquisition as material costs. Its hand-written AVX backend ranges from slower
than the generic loop to only marginally faster, and changing one per-size
instantiation perturbs neighbouring code generation. Continuing to add
size-specific Stockham schedules is therefore not an independently selectable
routing policy.

## Decision

Use one shared four-step selection function for every power-of-two caller, with
the measured crossover supplied by the workload that invokes it.

1. The general mixed-radix dispatcher retains `FOUR_STEP_THRESHOLD = 4096`.
   One-dimensional `pot_inplace` selects four-step at 65536, the point where
   its row transforms also enter the parallel Moirai route. Both callers share
   the even-exponent condition, split, execution, and normalization code.
2. The batched four-step driver reuses the authoritative cached
   `W_N^(j*k)` matrix. It does not evaluate `N` trigonometric functions on each
   transform call.
3. One-dimensional lengths below 65536 and asymmetric power-of-two splits
   retain Stockham. Normalized inverse execution applies normalization after
   four-step, matching the existing dispatcher contract.
4. Apollo pins Moirai merge `10082209`, whose indexed scopes borrow stack state
   instead of allocating `Arc` state per call. The parallel four-step therefore
   preserves the warm complex zero-allocation contract.

This decision changes internal routing only. The public transform API,
normalization convention, scratch ownership, and scalar support remain
unchanged.

## Rejected alternatives

### Continue specializing the hand-written AVX Stockham backend

Rejected because the measured backend loses to the generic loop at three of
four audited sizes, and prior per-size routing changed code generation outside
the selected size. Another isolated schedule would repeat a mechanism already
falsified without addressing the existing four-step reach discrepancy.

### Clone PhastFT's planar in-place kernel family

Rejected because Apollo's measured planar prototypes were slower than the
interleaved incumbent, including an explicit-SIMD variant. A separate kernel
family would duplicate the algorithm and enlarge the verification surface
without evidence that layout is the binding constraint.

### Keep four-step scoped to multidimensional axes

Rejected because the batched driver was introduced to close the power-of-two
throughput band and its mathematical decomposition applies to the same 1-D
transform. Leaving it unreachable would retain two routing policies for one
operation and leave the measured census on the known slower route.

### Use the general dispatcher's 4096 crossover for 1-D

Rejected by direct measurement in `benches/engine_census` (named here because
the original record did not name it, which is what made the figure
unfalsifiable; see the third revision note). Relative to the retained Stockham
entry run, selecting four-step at 4096 moved the 4096 median from 57.477 us to
338.85 us
and the 16384 median from 149.512 us to 1.6165 ms. The same route became
profitable at 65536 (920.95 us to 579.45 us) and 262144 (6.51215 ms to
2.4801 ms). One selector remains authoritative, but the caller supplies the
measured workload crossover rather than conflating axis and standalone costs.

### Extend the single-threaded batched driver above its current domain

Rejected because routing the high-size 1-D cases through that implementation
measured 7.314 ms at 65536 and 25.979 ms at 262144. The retained generic
four-step distributes independent rows through Moirai instead.

### Disable parallel rows to avoid scheduler allocation

Rejected because a serial generic four-step measured 7.663 ms at 65536 and
27.554 ms at 262144. The allocation belonged to Moirai's indexed scope state,
not to the FFT algorithm; fixing that provider removed the allocation without
discarding parallelism.

## Correctness and performance contract

For `N = m^2`, the driver computes the Cooley-Tukey factorization as `m`
length-`m` transforms, multiplication by `W_N^(j*k)`, a square transpose, and
the second `m` length-`m` transforms. Reusing the cached matrix changes only
where the same twiddle values come from; the analytical accuracy-growth gate
proves that the cache uses direct evaluation rather than recurrence.

Warm complex execution must remain allocation-free. The two real planes reuse
the existing `N`-complex scratch allocation, and cache hits clone `Arc`
handles without allocating. The decision-run census pinned to Moirai `10082209`
reports zero complex allocations at all five sizes and exactly one real-output
allocation of `16N` bytes.

Paired decision-run medians in nanoseconds, with the benchmark's median
confidence interval:

| N | Apollo | RustFFT | PhastFT |
| ---: | ---: | ---: | ---: |
| 1024 | 15309 [15266, 15348] | 1473 [1468, 1475] | 1673 [1670, 1676] |
| 4096 | 64835 [64730, 64990] | 8311 [8294, 8325] | 8790 [8775, 8824] |
| 16384 | 288950 [288350, 290250] | 38641 [38450, 38900] | 51312 [51116, 51558] |
| 65536 | 512950 [482300, 559600] | 466325 [463500, 471400] | 259283 [257500, 261366] |
| 262144 | 2870250 [2783000, 2942300] | 2492000 [2460800, 2545200] | 1306550 [1292600, 1333200] |

Apollo remains 10.0% slower than RustFFT at 65536 and 15.2% slower at 262144;
PhastFT remains 1.98x and 2.20x faster. At 1024 through 16384 the unchanged
Stockham route remains 7.5x to 10.4x behind the faster reference and is the
next CPU kernel target. The benchmark body completed in 3.10 seconds against
its 60-second bound.

The implementation is immutable at `5ca9deb4`. From a standalone checkout at
that revision, outside the Atlas development overlay, the confirmation command
is:

```text
cargo bench --locked --offline -p apollo-fft --bench engine_census
```

Two exact-commit repeats completed their benchmark bodies in 3.14 and 3.12
seconds. Both retained zero warm complex allocations and one `16N`-byte real
allocation. Their 65536 medians were 718.100 us [680.100, 756.400] and
615.550 us [572.900, 636.900]; their 262144 medians were 2.57070 ms
[2.51710, 2.66550] and 2.58700 ms [2.44420, 2.70240]. The cross-run
wall-clock intervals do not all overlap, so uncontrolled-host wall time remains
diagnostic rather than a deterministic regression gate. The route decision
rests on the paired entry/candidate experiment; the allocation contract is
stable across all three runs.

## Failure modes and verification

- A stale route that bypasses four-step fails a path-selection test at the
  threshold.
- Incorrect twiddle provenance fails the two-dimensional error-growth gate.
- Index, transpose, sign, or normalization defects fail direct-DFT, round-trip,
  and PhastFT differential tests for forward and inverse transforms.
- A cold-cache allocation mistaken for steady-state cost is excluded by one
  warm-up call before the allocation counter is enabled.
- A throughput regression blocks the route at the losing size; benchmark
  workload, sample count, and confidence rule remain unchanged.

## Revision note

2026-08-26: recorded after PR 121 proved that the batched path was unreachable
from 1-D and added executable coverage for both four-step implementations.

2026-08-26: revised after crossover experiments rejected a universal 4096
threshold, selected the parallel 65536 route for 1-D, and the Moirai provider
fix restored zero-allocation parallel execution.

2026-08-26 (third): the decision is unchanged and the reasoning behind it is
now recorded. The defect this revision fixes is that the crossover was stated
without naming the instrument that produced it, which made the figure
unfalsifiable and cost a full re-derivation to recover.

**The record now names its instrument.** `benches/engine_census`, which flushes
64 MiB between arms and measures Apollo against RustFFT, PhastFT and RealFFT in
one process, produced the rejected-alternative figures above. Re-run against
this revision's code they reproduce closely: selecting four-step at 4096 gives
348 us at N = 4096 against the recorded 338.85 us, and 1.64 ms at N = 16384
against the recorded 1.6165 ms.

**A second instrument disagreed, and both are right.** `pot::crossover` runs
both routes at one length in one process, with the cache flushed before each arm
and the arm order alternating so neither route is charged for reloading its own
input. It puts four-step ahead of Stockham from N = 256 upward and by 2 to 3x
through the whole ladder:

| N | stockham | four-step | ratio |
| --- | --- | --- | --- |
| 16 | 300 ns | 1800 ns | 6.00 |
| 64 | 1300 ns | 2100 ns | 1.62 |
| 256 | 5700 ns | 3300 ns | 0.58 |
| 4096 | 68500 ns | 23100 ns | 0.34 |
| 16384 | 296300 ns | 94400 ns | 0.32 |
| 262144 | 5324000 ns | 2669300 ns | 0.50 |

Reaching the same route through `FftPlan1D` rather than calling it directly
costs nothing measurable (23200 ns against 23500 at N = 4096), so the gap is not
plan overhead.

**The difference is the process, not the harness.** Timing `four_step_fft` from
inside the census binary shows it genuinely taking 99 us per call at N = 4096 —
a minimum over thousands of calls — where the same binary's test process takes
12. The twiddle matrix is built once in both (verified by counting builds: four,
one per size), and neither allocates per call. The route is not mismeasured
there; it is slower there.

The available explanation is layout: four-step holds three `N`-sized arrays live
at once — the data, the scratch, and the `W_N^(j*k)` matrix — against Stockham's
two, and how those three land relative to one another depends on allocation
history, which differs between a process holding four engines' plans and a
64 MiB flush buffer and a process holding one plan. That is a hypothesis with a
mechanism, not a measurement, and it is filed as
`ATLAS-APOLLO-FOUR-STEP-LAYOUT-SENSITIVITY-2026-08-26`.

**Why the threshold stays at 65536.** Between an isolated figure and one taken in
a process that resembles a caller, the second decides. Lowering the threshold on
the isolated measurement would ship a 12x regression at N = 4096 in exactly the
benchmark that represents real use. The isolated figure is not discarded — it
says the route itself is faster and that something around it is not, which is a
more useful statement than either number alone.

**What this host cannot settle.** Apollo's own N = 4096 Stockham figure moved
between 29 us and 65 us across runs in one session with its code untouched, so
the absolute values here bound nothing. What survives is reproducible within a
run: the ordering between routes, the ratio between them, and the 15x gap
between processes. Confirming the crossover on a quiet host remains open under
`ATLAS-APOLLO-CROSSOVER-REDERIVE-2026-08-26`.

**Structural change that came with this.** Routes are now zero-sized types
implementing `PotRoute` (`kernel/pot/route.rs`) rather than a bare `if` against
a constant. That is what let both routes run at one length in one process, which
is what made the two instruments comparable at all; admission is defined once on
`FourStep::admits`, so the general dispatcher and one-dimensional plans cannot
drift apart on which lengths the split is valid for. Selection remains one
branch per transform and the types carry no data.

2026-08-26 (fourth): the threshold moves to 4096, and the third revision's
open question — why the same call cost 12 us in one process and 99 us in
another — is answered: **the hybrid scheduler, not the code and not memory
layout.** The host is a Core Ultra 9 285K (8 P-cores, 16 E-cores). Windows
hands benchmark child processes EcoQoS — efficiency cores at efficiency
frequency — and instrumenting the census process showed the batched kernel
executing exclusively on E-cores (CPUs 8 through 21, wandering), every call
slow, while the identical binary elsewhere ran unthrottled.

`pot::core_matrix`, which pins the thread and so removes the scheduler from
the question, gives at N = 4096:

| route | P-core | E-core |
| --- | --- | --- |
| Stockham | 28.1 us | 62.6 us |
| four-step | **16.6 us** | **13.2 us** |

Four-step wins on both core types, consistent with `pot::crossover`'s
in-process ladder (ahead from N = 256 through 2^20). The third revision kept
65536 on the census's evidence; that evidence is now known to have measured
scheduling, so the constant follows the controlled instruments instead.

Plane-stride padding (`ROW_PAD`) was implemented while testing the layout
hypothesis and is kept on its own merits — +10% pinned on a P-core at 4096,
and it removes a real power-of-two aliasing hazard the fused transpose had to
tile around — but it did not and could not cure the anomaly, because the
anomaly was never layout.

The census now opts itself out of power throttling
(`PROCESS_POWER_THROTTLING_EXECUTION_SPEED`), which is necessary but not
sufficient on a contended host: absolute census figures from this machine
remain unusable while other work runs, and the quiet-host item stands. The
instruments to run there are `pot::crossover` and `pot::core_matrix`, both
named here so this figure is falsifiable in a way the original was not.
