# ADR 0011: Native benchmark measurement runtime

## Status

Accepted, 2026-07-15.

## Context

Criterion was Apollo's sole remaining transitive Rayon dependency. Its public
benchmark DSL also owned only timing orchestration; Apollo's seven benchmark
binaries already own the production closures, inputs, setup, and mathematical
workloads they measure.

## Decision

`apollo-bench` is the canonical benchmark measurement boundary. A benchmark
case carries its group, operation, and parameter. The runner performs a
sequential warm-up, estimates an integral batch size for the requested total
measurement budget, collects 100 normalized samples, and reports their minimum
and central-pair median as CSV.

Measured closures remain sequential. Moirai is the provider for transform
parallelism; scheduling independent cases concurrently would overlap timing
intervals and violate the per-operation measurement contract.

## Estimator theorem

For the fixed 100 samples `x₁ ≤ … ≤ x₂m`, where `m = 50`, the reported integer
nanosecond median is `⌊(xₘ + x₍m₊₁₎) / 2⌋`. At least `m` samples are at or below
`xₘ`, and at least `m` samples are at or above `x₍m₊₁₎`. Therefore fewer than
`m` arbitrarily large delay outliers cannot replace the central pair that
determines the reported statistic.

This is an elementary cardinality proof. It does not prove a timing result is
noise-free, portable across machines, or a comparison with an older Criterion
result. Performance claims still require an observed baseline comparison.

## Consequences

- The workspace has no Criterion or Rayon dependency edge.
- Benchmark source uses direct case execution rather than a Criterion-shaped
  compatibility API.
- Production closures, setup, and parameter matrices remain visible at their
  benchmark call sites. Standard budgets are centralized in
  `BenchmarkConfig::standard()`; case-specific nonstandard budgets remain
  explicit at their call sites.
- `cargo bench --no-run` verifies every benchmark binary without executing a
  timed workload or requiring a GPU device.
