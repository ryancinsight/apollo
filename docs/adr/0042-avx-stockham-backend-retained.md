# 0042 — The AVX Stockham backend is retained; f64 N = 256/512 route scalar

- Status: Accepted (2026-08-27)
- Item: `ATLAS-APOLLO-AVX-STOCKHAM-AUDIT-2026-08-25`
- Driving evidence: `backend_matrix` pinned probe (this change); entry finding
  in `gap_audit.md#pot-f64-profile`

## Context

The audit item carried a measured argument against the hand-written AVX
Stockham backend: scalar/AVX ratios of 0.35x at N = 256, 0.55x at 512, 1.11x
at 1024, 0.69x at 4096 — the AVX backend slower at three of four sizes — and
proposed retiring the largest part of `apollo-fft`'s `core::arch` surface on
that basis. Those numbers predate the repository's pinned instruments.

## Finding

The `backend_matrix` probe (same binary, both backends instantiated under
`cfg(test)`, interleaved, thread pinned per core type, Core Ultra 9 285K)
measures, clone-inclusive per call:

| n | f64 scalar P | f64 avx P | s/a P | s/a E | f32 s/a P | f32 s/a E |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 985 | 674 | 1.46 | 0.43 | 2.80 | 0.90 |
| 256 | 1386 | 1407 | 0.99 | 0.27 | 1.76 | 0.49 |
| 512 | 2614 | 3116 | 0.84 | 0.23 | 2.18 | 0.61 |
| 1024 | 9615 | 6272 | 1.53 | 0.45 | 2.88 | 0.99 |
| 2048 | 21106 | 14540 | 1.45 | 0.34 | 1.98 | 0.60 |
| 4096 | 43523 | 30952 | 1.41 | 0.34 | 1.40 | 0.51 |
| 8192 | 105653 | 74763 | 1.41 | 0.47 | 1.56 | 0.78 |
| 16384 | 199113 | 125888 | 1.58 | 0.41 | 1.38 | 0.54 |
| 32768 | 412400 | 279613 | 1.48 | 0.42 | 2.70 | 0.87 |

`s/a` is scalar-time over AVX-time: above 1.0 the AVX backend wins.

Two conclusions the entry numbers could not show:

1. **The retirement premise was a scheduler artifact.** On a pinned P-core the
   AVX backend wins nearly everywhere, for both precisions. The entry ratios
   (0.27–0.69 scalar/AVX) match the pinned E-core column, not the P-core one:
   the 2026-08-25 measurement ran unpinned and was EcoQoS-scheduled onto
   E-cores, the exact confound `pot::core_matrix` was later built to remove.
2. **f64 N = 256 and 512 are the exception on both core types.** At those two
   sizes the auto-vectorized scalar stages meet (256, P) or beat (512:
   −16% P, −77% E; 256: −73% E) the AVX stages regardless of where the
   scheduler lands the thread. No other size has a core-type-independent
   ordering.

## Decision

- **Retain the AVX Stockham backend** for both precisions. Retirement is
  rejected: it would regress every f32 size 1.4–2.9x and every f64 size except
  256/512 by 1.4–1.6x on P-cores.
- **Route f64 N = 256 and 512 through the scalar stages** in
  `StockhamKernel for f64` (both entry points). `PreciseStockham` becomes
  unconditionally compiled to serve this route. `ReducedStockham` stays
  test/non-AVX-gated; f32 keeps the AVX backend at every size.
- **Correct the record**: the audit item's retirement framing closes as
  premise-falsified-by-pinned-measurement; `gap_audit.md#pot-f64-profile`'s
  "hand-written AVX backend is slower than the auto-vectorized scalar" claim
  holds only under E-core scheduling.

## Alternatives rejected

- **Wholesale retirement** (the item's motivating option): rejected on the
  P-core column above.
- **Per-core dynamic backend selection**: production transforms cannot know
  where the scheduler will land the thread mid-call, and pinning is not the
  library's decision. Only both-core-dominant reroutes are safe; N = 256/512
  f64 are the only such sizes.
- **Rebuild on the planar layout**: premise closed by
  `ATLAS-APOLLO-POT-PLANAR-2026-08-25` (planar prototype worse at every size
  ≥ 2^10).

## Failure modes and limits

- The AVX-512 arms (`PreciseStockhamAvx512`) are unmeasurable on this host
  (Arrow Lake has no AVX-512) and are untouched; their evidence remains gated
  on real silicon (hermes HS-429 class).
- The scalar route at 256/512 autovectorizes at the build's baseline ISA
  (SSE2 here); a build with `-C target-feature=+avx2` would change both arms
  and the crossover should be re-measured there.
- E-core numbers say the AVX backend pays 2–4x under efficiency scheduling;
  routing cannot exploit this without core detection, so the cost is accepted
  and recorded rather than masked.

## Verification

- `backend_matrix` pinned probe: the table above, plus a numerical-agreement
  guard between backends bounded by rounding-difference growth over
  `log2 n` stages.
- Full `apollo-fft` nextest suite green at the change (scalar backend is
  covered by the same value-semantic suite that covers the AVX one).
- `rustfft_comparison` regenerated after the reroute; acceptance is no size
  regressed beyond host noise and the f64 512 row improved.
