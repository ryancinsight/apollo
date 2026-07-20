# ADR 0036: Native benchmark regression oracle

- **Status:** Accepted
- **Date:** 2026-07-20
- **Class:** [minor] [arch]

## Context

Apollo benchmarks use `apollo-bench`, a native sequential measurement runtime,
not Criterion. The added CI job ran the candidate once, copied that report as
its baseline, and compared the report to itself through a Python script copied
across repositories. This could not detect a change. Its workspace
`--all-features` command also pulled CUDA build requirements into a CPU
measurement lane.

The comparison must distinguish sampling noise from a supported slowdown
without introducing a second measurement provider or a second statistical
implementation.

## Decision

Keep report generation and interpretation in `apollo-bench`.

1. Extend each 100-sample CSV record with a symmetric, distribution-free
   confidence interval for the population median.
2. Discover CSV reports recursively and require identical report and case
   sets between independently executed base and candidate trees.
3. Require each interval to carry at least 95% coverage.
4. Counterbalance execution as baseline→candidate then candidate→baseline.
5. Classify a regression only when the candidate lower bound exceeds the
   baseline upper bound in both execution orders.
6. Delete the copied Python comparator. CI orchestration checks out and runs
   base and candidate revisions separately after the new schema reaches the
   default branch.

The strongest rejected alternative is converting Apollo to Criterion solely
to reuse the Atlas Criterion comparator. Apollo already owns a cohesive native
runtime used by its transform benchmarks; replacing that provider would widen
the change without improving the statistical contract.

## Mathematical contract

For ordered independent samples `X_(1), …, X_(n)`, the interval

\[
  [X_{(k)}, X_{(n-k+1)}]
\]

covers the population median with probability

\[
  1 - 2 P(\operatorname{Bin}(n, 0.5) \le k - 1).
\]

This is the distribution-free interval in
[NIST Technical Note 2119, section 5.3, equations 30–31](https://doi.org/10.6028/NIST.TN.2119).
For Apollo's fixed `n = 100`, the narrowest symmetric interval meeting 95%
coverage is `[X_(40), X_(61)]`; its exact coverage floors to 964799 parts per
million. Integer binomial counts encode this contract without floating-point
rounding.

The comparison makes no cross-machine performance claim. Base and candidate
must execute on the same hosted runner in one job. Hosted run `29757554816`
falsified a single fixed-order pair: source-identical revisions produced 31
disjoint candidate slowdowns, including one-nanosecond separations. Reversing
the order supplies the control for systematic thermal, frequency, and runner
drift. A slowdown must reproduce in both orders; otherwise it is order-sensitive
evidence, not a code-regression claim.

## Consequences

The CSV schema is additive and `apollo-bench` exposes an additive public
comparison API and CLI. Missing, malformed, low-confidence, or unpaired
evidence fails closed, including mismatched case universes across execution
orders. The base/head CI increment cannot precede this schema on the default
branch because legacy baseline reports do not contain the required intervals.
