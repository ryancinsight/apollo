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
and median as CSV.

Measured closures remain sequential. Moirai is the provider for transform
parallelism; scheduling independent cases concurrently would overlap timing
intervals and violate the per-operation measurement contract.

## Estimator theorem

For sorted samples `x₁ ≤ … ≤ x₂m₊₁`, the reported median `xₘ₊₁` has at least
`m + 1` samples at or below it and at least `m + 1` samples at or above it.
Therefore fewer than `m + 1` arbitrarily large delay outliers cannot move the
median outside the central order statistic.

This is an elementary cardinality proof. It does not prove a timing result is
noise-free, portable across machines, or a comparison with an older Criterion
result. Performance claims still require an observed baseline comparison.

## Consequences

- The workspace has no Criterion or Rayon dependency edge.
- Benchmark source uses direct case execution rather than a Criterion-shaped
  compatibility API.
- Existing warm-up and measurement budgets, production closures, setup, and
  parameter matrices remain visible at their benchmark call sites.
- `cargo bench --no-run` verifies every benchmark binary without executing a
  timed workload or requiring a GPU device.
