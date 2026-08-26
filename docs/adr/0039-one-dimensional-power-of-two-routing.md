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

Rejected by direct measurement. Relative to the retained Stockham entry run,
selecting four-step at 4096 moved the 4096 median from 57.477 us to 338.85 us
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
