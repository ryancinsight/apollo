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

1. Extend each 100-sample CSV record with its ordered observations and a
   symmetric, distribution-free descriptive interval for the population
   median.
2. Discover CSV reports recursively and require identical report and case
   sets between independently executed base and candidate trees.
3. At comparison time, derive simultaneous intervals whose individual
   miscoverage is at most `0.05 / (2m)` for `m` cases and two revisions.
4. Counterbalance execution as baseline→candidate then candidate→baseline.
5. Classify a regression only when the candidate lower bound exceeds the
   baseline upper bound in both execution orders.
6. Compile both revisions against the candidate `apollo-bench` source so the
   measurement instrument remains constant while the transform implementation
   varies.
7. Delete the copied Python comparator. CI orchestration checks out and runs
   base and candidate revisions separately after the new schema reaches the
   default branch.
8. Replicate ABBA with the phase-reversed BAAB schedule and require the same
   slowdown in all four comparisons.
9. Execute the hosted experiment only when a pull request changes the measured
   `apollo-fft` local dependency closure, the `apollo-bench` instrument, Cargo
   resolution, toolchain configuration, or the benchmark workflow itself.
   Keep this applicability boundary in a dedicated path-filtered workflow.

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
For Apollo's fixed `n = 100`, the narrowest symmetric individual interval
meeting 95% coverage is `[X_(40), X_(61)]`; its exact coverage floors to
964799 parts per million. A comparison over `m` cases derives a wider interval
with per-interval miscoverage no greater than `0.05 / (2m)`.
[Bonferroni's inequality](https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm)
therefore bounds the probability that any baseline or candidate interval
misses its population median by 5%, without requiring independence. Integer
binomial counts encode this contract without floating-point rounding.

The comparison makes no cross-machine performance claim. Base and candidate
must execute on the same hosted runner in one job. Hosted run `29757554816`
falsified a single fixed-order pair: source-identical revisions produced 31
disjoint candidate slowdowns, including one-nanosecond separations. Reversing
the order supplies the control for systematic thermal, frequency, and runner
drift. A slowdown must reproduce in both orders; otherwise it is order-sensitive
evidence, not a code-regression claim.

Hosted run `29759735814` falsified counterbalancing alone for a pull request
that changes `apollo-bench`: compiling each revision against its own harness
changed the measurement instrument as well as the code under test and produced
22 apparent regressions. CI therefore holds the candidate harness constant
across both revision builds and verifies that all benchmark entry points are
identical. Only the revision-specific transform implementation varies.

Hosted run `29761551514` held that instrument constant but still produced 25
apparent regressions. The comparator had applied a separate 95% interval to
each case without controlling the simultaneous comparison family. The report
therefore retains all ordered observations so the comparator can select the
exact family-size-dependent interval after it validates the full evidence
universe.

Hosted run `29764170548` applied the family-wise intervals but reported 12
slowdowns under one ABBA block despite an empty production transform diff
between base `66e37ab` and candidate `65dd9ad`. ABBA alone assigns the two
revisions to different periods of one runner timeline. Appending BAAB yields
baseline period positions `{1, 4, 6, 7}` and candidate positions
`{2, 3, 5, 8}`. Both sets sum to 18 and both squared sets sum to 102, so the
complete schedule balances revision exposure to constant, linear, and
quadratic period terms. The final
regression event is the intersection of the four family-wise comparison
events; it therefore remains bounded by 5% without assuming that the blocks
are independent.

## Consequences

The CSV schema carries the ordered observations as the statistical source of
truth, while the summary columns remain validated descriptive output.
`apollo-bench` exposes an additive public comparison API and CLI. Missing,
malformed, insufficient, or unpaired evidence fails closed, including
mismatched case universes across execution orders or replications. A pull
request that changes `apollo-bench` measures the base transform with the
candidate instrument; this intentionally evaluates transform regression
rather than benchmark-harness performance. The eight measurements roughly
double the empirical lane from 17 to 34 minutes while remaining inside its
60-minute purpose-specific bound. The base/head CI increment cannot precede
this schema on the default branch because legacy baseline reports do not
contain the ordered observations.

Exact-head hosted run `29766127266` passed the eight-run source-identical
canary and replicated comparison in 31 minutes. This validates the operational
orchestration on one hosted runner; it does not establish immunity to arbitrary
non-polynomial runner noise.

Later run `29788350487` supplied that overturning evidence: base `07462c0` and
candidate `b825fcb` had an empty diff over the complete measured source,
instrument, Cargo-resolution, and toolchain closure, but the hosted job still
reported two candidate slowdowns in all four comparisons. The smallest
separations were one to nine nanoseconds. A statistical gate cannot attribute
that source-identical variation to the release-only candidate. The experiment
therefore runs only for changes capable of altering its measured binary or
instrument. This changes the gate's applicability, not its workloads,
thresholds, sample count, or regression classifier.

Exact-head run `29790606838` passed the dedicated workflow's eight
measurements and replicated comparison in 31 minutes 38 seconds after the path
split. This validates the benchmark-relevant workflow path; path-selection
regressions establish release-only exclusion separately.

Hosted PR #64 run `29946182469` supplied a second source-identical
falsification. The base and merge candidate had identical production source,
manifest, lock, and toolchain inputs, but compiling them in separate absolute
checkout paths produced persistent f32 N=1031 automatic and forced-Bluestein
separations in all four comparisons. The experiment cannot attribute that
binary-level variation to production code.

The workflow therefore compiles baseline and candidate sequentially at one
canonical absolute path, copies each resulting executable before the next
revision occupies that path, and measures those immutable artifacts directly.
The candidate `apollo-bench` source and benchmark entry points remain pinned
into the baseline before compilation. SHA-256 identities are emitted as build
evidence; source-identical revisions can now reuse or reproduce the same
artifact rather than differing because their checkout paths differ.

The measurement workload uses geometric representatives for each distinct
dispatch regime instead of dense linear size sweeps. Every retained case still
records 100 ordered observations, and the family-wise interval and
phase-reversed ABBA/BAAB classifier are unchanged. The suite retains both f32
and f64 strategy comparisons and the f32 N=1031 Bluestein case that exposed
the false attribution. A 100 ms warm-up plus 400 ms measurement budget yields
approximately 21 seconds for `half_cyclic_rader`, 10 seconds for
`prime_compose`, and 11 seconds for `kernel_strategy` before process overhead.
Each binary has a 300-second hard bound. A full-case smoke mode uses minimum
timing budgets while retaining 100 samples per case under a 60-second bound.
These are instrument-design changes; no comparator threshold is widened and no
production transform path changes. On the reference Windows host the three
smoke runs complete in 0.75-0.81 seconds each; full measurement runs complete
in 9.44 seconds, 7.53 seconds, and 20.66 seconds respectively.
