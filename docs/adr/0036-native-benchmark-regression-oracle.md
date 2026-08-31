# ADR 0036: Native benchmark regression oracle

- **Status:** Accepted
- **Date:** 2026-07-20
- **Class:** [minor] [arch]
- **Revision 2026-08-27:** Smoke execution now invokes every unchanged case
  exactly once without warm-up or inferential statistics. Full measurement
  retains its budgets, 100 observations, and comparison contract. The change
  follows an unoptimized `cargo test --bench engine_census` runtime breach.

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

1. Extend each 100-sample CSV record with its ordered picosecond observations
   and a symmetric, distribution-free descriptive interval for the population
   median. Picosecond normalization preserves sub-nanosecond per-operation
   differences that integer nanoseconds would discard.
2. Discover CSV reports recursively and require identical report and case
   sets between independently executed base and candidate trees.
3. At comparison time, derive simultaneous intervals whose individual
   miscoverage is at most `0.05 / (2m)` for `m` cases and two revisions.
4. Counterbalance execution as baseline→candidate then candidate→baseline,
   with both revisions of each matched pair executing on the same hosted
   runner.
5. Classify a regression only when the candidate lower bound exceeds the
   baseline upper bound in both execution orders.
6. Compile both revisions against the candidate `apollo-bench` source so the
   measurement instrument remains constant while the transform implementation
   varies.
7. Delete the copied Python comparator. CI orchestration checks out and runs
   base and candidate revisions separately after the new schema reaches the
   default branch.
8. Replicate both orders twice as four independent matched-pair jobs and
   require the same slowdown in all four comparisons. The resulting evidence
   is equivalent to the phase-reversed ABBA/BAAB classifier but does not depend
   on one long runner timeline.
9. Execute the hosted experiment only when a pull request changes the measured
   `apollo-fft` local dependency closure, the `apollo-bench` instrument, Cargo
   resolution, toolchain configuration, or the benchmark workflow itself.
   Keep this applicability boundary in a dedicated path-filtered workflow.
10. Compile baseline and candidate artifacts concurrently at the same
    canonical checkout path on the pinned runner image. Compile only the three
    benchmark targets consumed by the gate, restore the shared compiler cache,
    and pass immutable executables and reports between jobs as one-day
    artifacts.

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

The comparison makes no cross-machine absolute-performance claim. Base and
candidate must execute on the same hosted runner within each matched pair.
Different pairs may use different hosts because the classifier first requires
a slowdown within every pair and then charges the result the complete
cross-pair spread: the slowest baseline upper bound must remain below the
fastest candidate lower bound. Host heterogeneity can suppress evidence but
cannot manufacture this final separation. Hosted run `29757554816`
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
`{2, 3, 5, 8}`. Both sets sum to 18 and both squared sets sum to 102, so that
historical one-runner schedule balanced revision exposure to constant, linear,
and quadratic period terms. The current topology instead executes those four
ordered pairs independently. The final regression event remains the
intersection of the four family-wise comparison events and therefore stays
bounded by 5% without assuming that the pairs are independent.

## Consequences

The CSV schema carries ordered integer-picosecond observations as the
statistical source of truth, while the summary columns remain validated
descriptive output.
`apollo-bench` exposes an additive public comparison API and CLI. Missing,
malformed, insufficient, or unpaired evidence fails closed, including
mismatched case universes across execution orders or replications. A pull
request that changes `apollo-bench` measures the base transform with the
candidate instrument; this intentionally evaluates transform regression
rather than benchmark-harness performance. The initial serialized
implementation's eight measurements roughly doubled the empirical lane from
17 to 34 minutes while remaining inside its 60-minute purpose-specific bound.
The base/head CI increment cannot precede this schema on the default branch
because legacy baseline reports do not contain the ordered observations.

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

The workflow therefore compiles baseline and candidate concurrently in
separate jobs that use the same canonical absolute checkout path, then measures
the immutable artifacts directly.
The candidate `apollo-bench` source and benchmark entry points remain pinned
into the baseline before compilation. SHA-256 identities are emitted as build
evidence; source-identical revisions can now reuse or reproduce the same
artifact rather than differing because their checkout paths differ. When all
three executable pairs are byte-identical, binary identity is conclusive that
the candidate cannot cause a performance regression, so the empirical
comparison is inapplicable. Differing executable pairs retain the complete
replicated measurement and comparison path.

The measurement workload uses geometric representatives for each distinct
dispatch regime instead of dense linear size sweeps. Every retained case still
records 100 ordered observations, and the family-wise interval plus four
ordered-pair acceptance rule are unchanged. The suite retains both f32 and f64
strategy comparisons and the f32 N=1031 Bluestein case that exposed the false
attribution. A 100 ms warm-up plus 400 ms measurement budget yields
approximately 21 seconds for `half_cyclic_rader`, 10 seconds for
`prime_compose`, and 11 seconds for `kernel_strategy` before process overhead.
Each binary has a 300-second hard bound. A full-case smoke mode invokes each
production closure once without warm-up and marks its single observation as
non-inferential. The committed bench gate applies a 60-second process bound.
Every benchmark executable reads `APOLLO_BENCH_MODE` before constructing its
suite; using `BenchmarkSuite::default()` would silently select the full
measurement budget and violates the smoke contract.
GPU preparation comparisons retain one largest representative per dimensional
regime when a single preparation observation takes hundreds of milliseconds.
This keeps the established 100-observation estimator while avoiding redundant
sizes whose dispatch strategy and oversampled grid are identical.
These are instrument-design changes; full measurement retains 100 observations
per case, no comparator threshold is widened, and no production transform path
changes. Before the one-observation correction, the reference Windows host ran
the three minimum-budget smoke binaries in 0.75-0.81 seconds each; full
measurement runs complete in 9.44 seconds, 7.53 seconds, and 20.66 seconds
respectively. After the correction, all seven custom bench binaries complete
their unoptimized smoke gate in 26.8 seconds after linking, while the unchanged
100-observation optimized engine census completes in 5.57 seconds.

Hosted run `29955865616` confirmed that canonical-path compilation produced
byte-identical SHA-256 values for every base/candidate executable. It then
falsified empirical comparison of identical binaries by labeling one side
candidate: `composite_radix_order/r4_2_5_5_f64/200` separated by 1-7 ns in all
four comparisons. Binary identity is stronger causal evidence than sampled
timing for this boundary; the identity exit prevents arbitrary labels from
turning runner noise into a production-regression claim.

Exact-head hosted run `29956621276` validated the final decision path. It
compiled both revisions at the canonical path, passed all three smoke
executions, proved all three executable pairs byte-identical, and accepted the
identity evidence in 4 minutes. Exact-head CI run `29956621235` independently
passed the Rust workspace and Python binding jobs. This is static causal
evidence for the unchanged artifacts, not empirical performance evidence; any
differing executable pair enters the complete four-matched-pair experiment.

## Revision: 2026-08-26

The post-PR #127 hosted run completed in 12 minutes 36 seconds: 7 minutes
11 seconds compiled seven FFT benchmark targets even though the gate retained
three, and 5 minutes 4 seconds serialized all eight complete measurements.
It also exposed integer-nanosecond normalization as a resolution defect for
sub-2 ns kernels. The workflow now compiles only the consumed targets, restores
compiler artifacts, builds baseline and candidate concurrently at the same
canonical path, and executes the four matched pairs concurrently. Each pair
still keeps both revisions on one runner in the prescribed order, and the
comparator, 100 observations, confidence construction, workloads, and
four-comparison acceptance rule are unchanged. The operational critical path
is therefore one artifact build plus one matched pair instead of two builds
plus eight serialized measurements; hosted validation of the resulting bound
remains the delivery gate for this revision.

## Revision: 2026-08-31

The default `bench` profile had inherited release and then overridden its
single codegen unit with eight. This contradicted the release profile's stated
cross-module inlining and vectorization requirement. The unchanged QFT route
instrument reproduced a 42.745-microsecond N=1,024 median under the divergent
profile and a 1.955-microsecond median under `--profile release`; the release
pinned route probe independently measured 1.819--3.036 microseconds across the
host's core classes. The 21.86x separation falsified a production routing
defect and identified benchmark-profile drift.

The default benchmark profile now inherits release without optimization or
codegen overrides. `bench-quick` remains the explicit profile for smoke runs
that trade code quality for compilation latency. Benchmark bodies, workloads,
sample counts, timing regions, and the production FFT route are unchanged.
The corrected default profile measures the unchanged N=1,024 QFT case at
1.954 microseconds.

The bounded RustFFT comparison additionally binds one exact logical processor
through Hermes for the complete measurement process. An explicit
`APOLLO_BENCH_PROCESSOR` selects the processor; otherwise a supported host binds
the processor observed at startup. Unsupported hosts emit an unpinned warning,
so those runs remain smoke or correctness evidence rather than timing evidence.
Binding, parsing, and post-bind identity failures remain typed errors.

Comparator lifecycle is part of the measurement contract. Both engines build
plans and reusable scratch before timing; each timed closure restores identical
input and executes one transform. RustFFT's convenience `process` method is
rejected because it allocates and zero-fills scratch on every call. The
comparison uses `process_with_scratch` with the planner-reported scratch size.
Keeping the convenience call was the strongest rejected alternative: it would
measure allocator and initialization cost for one engine against retained
execution for the other, not transform throughput.

The representative default set now includes 1,024, 2,048, and 32,768 so the
instrument observes Apollo's large fixed-length specialization and its adjacent
control. Four 100-sample processor-2 runs complete in 6.93--6.97 seconds.
Every added row stays within 5.0% across the four runs except Apollo f64 at
32,768, which spans 126.293--154.510 microseconds while its RustFFT control
spans 129.064--132.700 microseconds. Exact binding controls processor identity,
not frequency state or interruption. The comparison therefore preserves full
sample distributions and remains one-run descriptive evidence; it does not
claim cross-run or cross-machine invariance.
