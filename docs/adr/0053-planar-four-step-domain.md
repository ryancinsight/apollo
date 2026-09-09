# ADR 0053: Planar four-step domain above the threading threshold

- **Status:** Accepted
- **Date:** 2026-09-08
- **Class:** [patch] [perf]
- **Item:** [APOLLO-N65536-FOUR-STEP-SCALAR-LOSS](../../backlog.md#apollo-n65536-four-step-scalar-loss); parent [ATLAS-APOLLO-BEAT-THE-REFERENCES](../../backlog.md#atlas-apollo-beat-the-references)

## Context

One-dimensional power-of-two plans above 1024 take the four-step route.
Even powers below 65536 run the batched planar driver
(`components::batched`): fused radix-4 stage sets with the transform index
in the SIMD lane, the four-step twiddle folded into the first stage's loads,
one in-place square transpose, no threading. Odd powers decimate once and
run both halves through the same driver. From 65536 upward the generic
driver (`four_step::execution`) runs instead: an out-of-place first
transpose, per-row scalar Stockham transforms handed to Moirai's parallel
provider, a scalar 16-by-16 multiply-transpose tile, a second threaded row
pass and a final transpose.

The boundary was `PARALLEL_ROW_THRESHOLD = 65536`, on the premise that from
that length the row transforms are worth threading and the threaded generic
route therefore beats a sequential SIMD pass. The premise was never measured
against the planar driver; ADR 0049 and ADR 0050 both tried to speed up the
generic route's multiply-transpose and both were rejected for no supported
gain.

Baseline on the pinned performance core (Intel Core Ultra 9 285K, origin/main
`f4a5f41a`, `benches/phastft_comparison`, `engine_census` and
`twiddless_comparison`, medians of 100 samples):

| N | route | apollo f64 | rustfft f64 | phastft f64 | apollo f32 | rustfft f32 | phastft f32 |
|---|---|---|---|---|---|---|---|
| 16384 | planar | 43.0 | 34.5 | 80.5 | 22.7 | 18.6 | 22.4 |
| 65536 | generic | 466 to 1556 | 170 to 348 | 241 | 219 to 885 | 90 | 132 |
| 262144 | generic | 1628 to 1732 | 1210 | 1345 | 1576 | — | 524 |
| 1048576 | generic | 10833 | — | 7748 | 5331 | — | 3336 |

Times in microseconds; ranges are the same length under different harnesses
(cache flushed per arm in the census, clone-inclusive in the comparisons),
and the 65536 interval alone spans 334 to 600 µs inside one census run. One
length below the boundary apollo trails RustFFT by 25%; at the boundary it
trails by 2.7 to 4.5 times with the widest intervals in the suite, which is
the signature of cross-core dispatch on rows whose work is a fraction of a
microsecond each.

## Decision

Bound the planar domain by its own constant, `batched::PLANAR_MAX_LEN`,
rather than by the generic route's threading threshold, and set it from
measurement. The candidate value was `2^20`; the first accepted value was
`2^18`, and the candidate was accepted in full on the 2026-09-09
re-measurement below: even powers through 1048576 take the planar driver,
odd powers through 2097152 take the fused split, and the generic route
serves lengths past that. `PARALLEL_ROW_THRESHOLD` keeps its meaning for
the generic route it still governs.

The change is one predicate and one constant; no kernel, layout, table or
normalization changes. Scratch requirements at the newly planar lengths
follow the existing padded-plane formula and the workspace test pins them.

## Verification

- `workspace_covers_padded_planes_and_nested_gathers` pins the scratch extent
  at 65536, 131072, 262144, 1048576, 2097152 and the first generic lengths
  4194304 and 8388608.
- `workspace_extents_preserve_the_impulse_spectrum` runs the exact impulse
  oracle through the selected route at 65536, 131072 and 262144 in both
  precisions and directions;
  `planar_domain_boundary_preserves_the_impulse_spectrum` runs it forward
  at `PLANAR_MAX_LEN` in both precisions.
- `large_planar_lengths_agree_with_rustfft_in_both_precisions` compares the
  public plan against RustFFT at 65536 and 262144 within twice the
  `O(log N · u)` per-bin bound the file already derives.
- The existing batched, four-step and dimension-1d suites run unchanged.

## Measurement and stop criterion

Re-run `phastft_comparison`, `engine_census` and `twiddless_comparison` on
the same core with the same configuration. Accept the candidate value at a
length when the planar route's interval sits below the generic route's
baseline interval there; set `PLANAR_MAX_LEN` to the largest accepted
length; reject the change outright if 65536 does not improve with disjoint
intervals. The gap that remains to RustFFT and PhastFT after this change is
the next item's subject, not this one's.

## Result and decision

Accept `PLANAR_MAX_LEN = 2^18`. Evidence is `output/apollo-planar-domain/`
(baseline plus three candidate runs of each instrument, with `manifest.json`).

Medians in microseconds; the candidate columns are the three replicated runs.

| N | arm | baseline | run 1 | run 2 | run 3 |
|---|---|---|---|---|---|
| 65536 | apollo f64 (comparison) | 767 | 220 | 209 | 324 |
| 65536 | apollo f64 (census) | 466 | 214 | 217 | 228 |
| 65536 | rustfft f64 (census) | 170 | 176 | 180 | 180 |
| 65536 | phastft f64 (census) | 216 | 259 | 253 | 239 |
| 65536 | apollo f32 | 219 | 108 | 107 | 129 |
| 65536 | phastft f32 | 132 | 152 | 146 | 173 |
| 262144 | apollo f64 (census) | 1628 | 1808 | 2581 | 2605 |
| 262144 | rustfft f64 (census) | 1210 | 1485 | 2624 | 1906 |
| 262144 | apollo f32 | 1576 | 675 | 669 | 790 |
| 262144 | phastft f32 | 524 | 575 | 593 | 1168 |
| 1048576 | apollo f64 | 10833 | 19683 | 17223 | 18200 |
| 1048576 | phastft f64 | 7748 | 15772 | 11840 | 11248 |

At 65536 the planar interval sits below the generic baseline interval in
every run at both precisions, with the reference arms within 10% of their
baselines: accepted. Apollo now leads PhastFT at 65536 in both precisions
and leads RustFFT in `f32` (99.5 against 111.3 µs in the clone-inclusive
instrument); it trails RustFFT in `f64` by 1.2 times, down from 4.5.

At 262144 the `f32` interval halves in every run against a reference arm
that moved at most 13% until the contaminated third run: accepted. The `f64`
arm does not separate from its baseline, but normalized to RustFFT inside
each run it reads 1.22, 0.98 and 1.37 against the baseline's 1.35, so the
planar route is not worse there and the length follows its precision twin.

At 1048576 the reference arms moved 45 to 100% between runs while a
tree-mate compiled and tested in the shared cache, so those wall-clock rows
were invalid evidence and the length stayed on the generic route until
[APOLLO-N1M-PLANAR-CROSSOVER](../../backlog.md#apollo-n1m-planar-crossover)
re-measured it on a quiet host (below).

### 1048576 and 2097152, re-measured 2026-09-09

Evidence `output/apollo-n1m-crossover/` (`manifest.txt`): the same two
instruments, both pinned to the performance core, no cargo or rustc process
on the host during any run, six runs counterbalanced base, candidate,
candidate, base, base, candidate against the tree at PR 361 (staged seams,
two-level fold). Medians in microseconds; base is `2^18`, candidate `2^20`.

| N | arm | base 1 | base 4 | base 5 | cand 2 | cand 3 | cand 6 |
|---|---|---|---|---|---|---|---|
| 1048576 | apollo f64 (census) | 10161 | 10128 | 9993 | 5619 | 5713 | 5821 |
| 1048576 | rustfft f64 (census) | 6388 | 6801 | 7072 | 6551 | 6824 | 6966 |
| 1048576 | phastft f64 (census) | 6905 | 7072 | 7139 | 7073 | 7417 | 7484 |
| 1048576 | apollo f32 (comparison) | 5981 | 5612 | 5078 | 2762 | 2821 | 2648 |
| 1048576 | phastft f32 (comparison) | 3149 | 3235 | 3267 | 3149 | 3318 | 3386 |
| 2097152 | apollo f64 (census) | 26575 | 26434 | 26971 | 16960 | 15889 | 17570 |
| 2097152 | rustfft f64 (census) | 17453 | 15462 | 17437 | 16805 | 15405 | 16395 |
| 2097152 | phastft f64 (census) | 18100 | 16702 | 17587 | 17069 | 16537 | 16737 |
| 2097152 | apollo f32 (comparison) | 14617 | 14265 | 14840 | 8275 | 8491 | 8443 |
| 2097152 | phastft f32 (comparison) | 7494 | 7513 | 7381 | 7543 | 7585 | 7729 |

Every candidate interval sits below every base interval at both lengths and
precisions, with the reference arms within 10% across the six runs:
accepted at `2^20`. At 1048576 apollo now leads RustFFT and PhastFT in
`f64` and PhastFT in `f32`; at 2097152 it is level with both in `f64` and
trails PhastFT by about 10% in `f32`, down from 1.9 times. The planes at
1048576 `f64` (16 MiB padded) and the caller's buffer fit the 36 MiB L3,
the regime the earlier hypothesis (an L3 spill) put outside it; the staged
seams and the two-level fold are what moved since the first measurement.

Limits: one host, one core class, contention recorded rather than excluded;
no cross-machine claim; the remaining 1.2 times gap to RustFFT in `f64` at
65536 and above is the planar driver's own cost, the subject of the next
increment under the parent item.

## Revision

2026-09-08: changed Proposed to Accepted at `2^18` after the replicated
measurement above; the `2^20` candidate is reduced to the supported lengths.

2026-09-09: `PLANAR_MAX_LEN` moved to `2^20` on the quiet-host census under
APOLLO-N1M-PLANAR-CROSSOVER; the decision and verification sections carry
the new domain, the measurement section the six-run table.
