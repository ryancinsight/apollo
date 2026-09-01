# 0042 — The AVX Stockham backend is retained pending investigation; f64 N = 256/512 route scalar

- Status: Accepted (2026-08-27)
- Item: `ATLAS-APOLLO-AVX-STOCKHAM-AUDIT-2026-08-25` (reopened 2026-09-01)
- Driving evidence: `backend_matrix` pinned probe, re-run 2026-09-01 on
  queried-class processors
- **Revision 2026-09-01: the finding this record was built on was inverted.**
  The original table's `P` column was measured on an efficiency core and its
  `E` column on a performance core, because the probe pinned to cpu 2 and
  cpu 12 and labelled by `landed < 8` while this host's performance set is
  `{0, 1, 10, 11, 12, 13, 22, 23}` (ADR
  [0043](0043-measurement-core-class-is-queried.md)). The 2026-08-27 record
  concluded "on a pinned P-core the AVX backend wins nearly everywhere"; the
  corrected measurement says the opposite. The prior text is not preserved
  here — git is the archive — but the decision it reached is explicitly
  revisited below, and the audit item it closed is reopened.

## Context

The audit item carried a measured argument against the hand-written AVX
Stockham backend: scalar/AVX ratios of 0.35x at N = 256, 0.55x at 512, 1.11x
at 1024, 0.69x at 4096 — the AVX backend slower at three of four sizes — and
proposed retiring the largest part of `apollo-fft`'s `core::arch` surface on
that basis.

The 2026-08-27 record rejected that premise as a scheduler artifact, on a
pinned table whose two columns turned out to be swapped. Corrected, the entry
ratios do not sit in an efficiency-core band at all: they sit close to the
*performance*-core column, which is what an unpinned thread mostly runs on.
The retirement premise was never falsified.

## Finding

`backend_matrix` re-run 2026-09-01 (same binary, both backends instantiated
under `cfg(test)`, interleaved, thread pinned to a processor whose class is
queried from `GetLogicalProcessorInformationEx`, Core Ultra 9 285K, release).
Three runs; the probe reports best-of-24 blocks per arm and the table gives the
median of the three run ratios with the observed range. Host was otherwise
loaded at 11–13% by concurrent agent sessions; the ratio is taken within a run
against the same cache state, and the run-to-run range below bounds what that
load contributed.

`s/a` is scalar-time over AVX-time: **above 1.0 the AVX backend wins, below 1.0
the auto-vectorized scalar backend wins.**

Performance core (cpu 1, queried class `performance`):

| n | f64 scalar ns | f64 avx ns | f64 s/a | range | f32 s/a | range |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 748.5 | 1844.6 | 0.405 | 0.405–0.406 | 0.894 | 0.887–0.898 |
| 256 | 1002.5 | 3703.0 | 0.271 | 0.269–0.276 | 0.485 | 0.484–0.486 |
| 512 | 1696.9 | 7527.3 | 0.225 | 0.224–0.236 | 0.583 | 0.582–0.584 |
| 1024 | 6738.7 | 15315.2 | 0.441 | 0.440–0.446 | 0.953 | 0.949–0.957 |
| 2048 | 11579.7 | 32396.9 | 0.356 | 0.341–0.358 | 0.588 | 0.587–0.593 |
| 4096 | 24050.0 | 66062.5 | 0.363 | 0.361–0.367 | 0.510 | 0.509–0.517 |
| 8192 | 77259.4 | 147268.8 | 0.528 | 0.496–0.534 | 0.794 | 0.771–0.799 |
| 16384 | 123018.8 | 293012.5 | 0.406 | 0.404–0.420 | 0.557 | 0.539–0.567 |
| 32768 | 253087.5 | 594750.0 | 0.426 | 0.402–0.445 | 0.892 | 0.884–0.946 |

Efficiency core (cpu 3, queried class `efficiency`):

| n | f64 scalar ns | f64 avx ns | f64 s/a | range | f32 s/a | range |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 983.3 | 676.1 | 1.452 | 1.439–1.458 | 2.852 | 2.832–2.852 |
| 256 | 1397.4 | 1408.0 | 0.989 | 0.988–0.996 | 1.764 | 1.759–1.778 |
| 512 | 2590.8 | 3126.4 | 0.831 | 0.829–0.836 | 2.159 | 2.149–2.162 |
| 1024 | 9075.8 | 6170.3 | 1.474 | 1.467–1.525 | 2.896 | 2.891–2.962 |
| 2048 | 18160.9 | 13589.8 | 1.330 | 1.325–1.347 | 1.910 | 1.899–2.016 |
| 4096 | 42437.5 | 35553.1 | 1.190 | 1.115–1.205 | 1.328 | 1.328–1.355 |
| 8192 | 110968.8 | 71396.9 | 1.554 | 1.410–1.588 | 1.520 | 1.506–1.989 |
| 16384 | 205712.5 | 127737.5 | 1.600 | 1.562–1.628 | 1.525 | 1.394–2.018 |
| 32768 | 415912.5 | 279125.0 | 1.512 | 1.484–1.591 | 3.412 | 3.227–3.413 |

The f32 efficiency-core ratios at 8192 and 16384 are the only unstable cells
(1.506–1.989 and 1.394–2.018); nothing below rests on them.

Three conclusions:

1. **The inversion is confirmed by reproduction, not only by the class query.**
   The corrected *efficiency*-core column reproduces the 2026-08-27 record's
   `P` column cell for cell — 128: 1.452 against the recorded 1.46; 256: 0.989
   against 0.99; 512: 0.831 against 0.84; 1024: 1.474 against 1.53 — on a
   different efficiency core (cpu 3, then cpu 7) from the one originally
   measured (cpu 2). The old `P` column was an efficiency core.
2. **The retirement premise is supported on performance cores.** At every
   probed size and both precisions, `s/a < 1.0` on the performance core: the
   auto-vectorized scalar backend beats the hand-written AVX backend by
   1.05–4.4x (f64 2.2–4.4x, f32 1.05–2.1x). This is what the audit item's entry
   measurement claimed and what the 2026-08-27 record rejected.
3. **The AVX backend is absolutely slower on a performance core than on an
   efficiency core, and this is unexplained.** At N = 512 f64 the AVX arm takes
   7527 ns on cpu 1 against 3126 ns on cpu 3 — 2.4x slower on the faster core —
   while the scalar arm behaves normally (1697 ns against 2591 ns, the
   performance core 1.53x faster as expected). Same binary, same inputs,
   interleaved in one run, reproduced across three runs and two distinct
   processor pairs. A backend that inverts the core hierarchy is showing a
   microarchitectural pathology, not simply losing to the optimizer, and no
   mechanism for it is established here.

## Decision

- **Retain the AVX Stockham backend for now, and reopen the audit.** The
  2026-08-27 grounds for retention — "it wins on P-cores" — are withdrawn as
  factually inverted. Retention is no longer a decision that the evidence
  supports; it is the status quo held while finding 3 is investigated, because
  retiring a backend on a measurement that inverts the core hierarchy would be
  acting on an unexplained result. `ATLAS-APOLLO-AVX-STOCKHAM-AUDIT-2026-08-25`
  reopens with finding 3 as its first step.
- **Route f64 N = 256 and 512 through the scalar stages** in
  `StockhamKernel for f64` (both entry points) — **unchanged and re-verified.**
  This reroute was justified as "scalar wins on both core types", which is
  label-independent, and the corrected measurement confirms it: at 256,
  `s/a` = 0.271 (performance) and 0.989 (efficiency); at 512, 0.225 and 0.831.
  Scalar wins on both classes at both sizes. `PreciseStockham` stays
  unconditionally compiled to serve this route.
- **No other production routing changes in this record.** The per-size picture
  is now core-class-dependent in the opposite direction from before — scalar
  ahead on performance cores, AVX ahead on efficiency cores at most sizes — and
  by this ADR's own standing rule only both-core-dominant reroutes are safe.
  Acting on the performance-core column alone would repeat the original error
  in mirror image.
- **Correct the record**: `gap_audit.md#stockham-backend-matrix` and the
  backlog verdict are corrected; the claim in `gap_audit.md#pot-f64-profile`
  that the hand-written AVX backend is slower than the auto-vectorized scalar
  one is **reinstated without the "unpinned scheduling only" qualifier** the
  2026-08-27 record attached to it.

## Alternatives rejected

- **Wholesale retirement** (the item's motivating option): not rejected any
  more, but not adopted here either. It is now the leading candidate, blocked
  on finding 3 and on f32 efficiency-core coverage, and is decided in the
  reopened item rather than in a revision note.
- **Per-core dynamic backend selection**: production transforms cannot know
  where the scheduler will land the thread mid-call, and pinning is not the
  library's decision. Only both-core-dominant reroutes are safe; N = 256/512
  f64 remain the only such sizes.
- **Rebuild on the planar layout**: premise closed by
  `ATLAS-APOLLO-POT-PLANAR-2026-08-25` (planar prototype worse at every size
  ≥ 2^10).

## Failure modes and limits

- **Probe coverage:** the matrix times the staged route (`transform_sized`) at
  128–32768. Unprobed but production-reachable AVX routes: the fixed-length
  32/64 leaves (reached standalone and as four-step rows at N = 1024/4096) and
  the 4096 four-triple special, which production reaches through `dispatch.rs`
  while the probe bypasses it — the table's 4096 row times the staged body, not
  that special. Neither route is changed by this decision.
- **Finding 3 bounds everything else.** Until the AVX arm's performance-core
  behaviour is explained, the performance-core column measures "this AVX
  implementation on this microarchitecture", not "hand-written AVX versus the
  optimizer" in general.
- **Blast radius of the reroute:** the scalar 256/512 branch also serves
  four-step inner rows (256-rows at N = 65536, 512-rows at N = 262144) and
  2-D/3-D lane transforms, which the `rustfft_comparison` default sweep
  (sizes ≤ 512, and standalone 256 routes four-step) cannot witness.
- The AVX-512 arms (`PreciseStockhamAvx512`) are unmeasurable on this host
  (Arrow Lake has no AVX-512) and are untouched; their evidence remains gated
  on real silicon (hermes HS-429 class).
- The scalar route at 256/512 autovectorizes at the build's baseline ISA
  (SSE2 here); a build with `-C target-feature=+avx2` would change both arms
  and the crossover should be re-measured there.
- **Host load:** 11–13% from concurrent agent sessions during the re-run. The
  per-run ranges above bound its contribution; the two unstable f32 cells are
  named rather than averaged away.

## Verification

- `backend_matrix` pinned probe, three runs on queried-class processors: the
  tables above, plus a numerical-agreement guard between backends bounded by
  rounding-difference growth over `log2 n` stages.
- Independent confirmation of the class assignment: the performance-core mask
  `0xc03c03` from `GetLogicalProcessorInformationEx`, and a separate per-processor
  timing sweep that partitions the host identically (ADR 0043).
- Cross-check against the superseded record: the corrected efficiency-core
  column reproduces the 2026-08-27 `P` column on a different efficiency core,
  which is what makes the relabelling a reproduction rather than a
  reinterpretation.
- The 2026-08-27 rejection of a `rustfft_comparison` regeneration is
  **withdrawn**: that run was rejected for reproducing "pinned E-core times",
  which were in fact ordinary performance-core times, so host load was not
  demonstrated and the run was not disqualified on the stated grounds.
