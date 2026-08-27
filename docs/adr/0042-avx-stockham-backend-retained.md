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

1. **The retirement premise does not survive pinning.** On a pinned P-core
   the AVX backend wins nearly everywhere, for both precisions. The entry
   ratios at 256/512/4096 (0.35/0.55/0.69 scalar/AVX) sit in the pinned
   E-core band (0.27/0.23/0.34), not the P-core one, and the 1024 entry
   (1.11) matches neither column — consistent with an unpinned thread
   migrating between core types mid-run. E-core (EcoQoS) scheduling is the
   leading hypothesis for the entry numbers, not an established identity:
   the two instruments differ in harness (clone-inclusive vs not), so the
   match is qualitative. What is established is that the pinned instrument
   reverses the entry ordering on the cores where throughput work runs.
2. **f64 N = 256 and 512 are the exception on both core types.** At those two
   sizes the auto-vectorized scalar stages meet (256, P) or beat (512:
   −16% P, −77% E; 256: −73% E) the AVX stages regardless of where the
   scheduler lands the thread. No other size has a core-type-independent
   ordering.

## Decision

- **Retain the AVX Stockham backend** for both precisions. Retirement is
  rejected: at every probed size of the staged route (128–32768) it would
  regress f32 by 1.4–2.9x and f64 by 1.4–1.6x on P-cores, 256/512 f64
  excepted.
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

- **Probe coverage:** the matrix times the staged route
  (`transform_sized`) at 128–32768. Unprobed but production-reachable AVX
  routes: the fixed-length 32/64 leaves (reached standalone and as four-step
  rows at N = 1024/4096) and the 4096 four-triple special, which production
  reaches through `dispatch.rs` while the probe bypasses it — the table's
  4096 row times the staged body, not that special. Neither route is changed
  by this decision, so the omissions bound the table's claims, not the diff's
  safety.
- **Blast radius of the reroute:** the scalar 256/512 branch also serves
  four-step inner rows (256-rows at N = 65536, 512-rows at N = 262144) and
  2-D/3-D lane transforms, which the `rustfft_comparison` default sweep
  (sizes ≤ 512, and standalone 256 routes four-step) cannot witness. The
  pinned per-call ordering covers those call shapes; the end-to-end check at
  large N rides the next idle-host census run.
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
- A `rustfft_comparison` regeneration was run and rejected as evidence: the
  host was loaded by a concurrent build and the run's apollo arms reproduce
  pinned E-core times (f64@128 1857 ns vs pinned E-core AVX 1834 ns). The
  committed table's f64@512 row (3123 ns) matches the pinned AVX arm
  (3116–3124 ns); the post-change scalar arm (2582–2614 ns) implies that row
  moves ~2.7x → ~2.3x, to be confirmed by the next idle-host regeneration.
  Note the default sweep caps at 512 and standalone 256 routes four-step, so
  it witnesses only the 512 row of this change either way.
