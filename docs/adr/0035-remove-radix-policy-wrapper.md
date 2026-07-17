# ADR 0035: Remove the Apollo radix-policy wrapper

- **Status:** Accepted
- **Date:** 2026-07-17
- **Class:** [major]

## Context

`apollo-fft` exposed `application::execution::policy::RadixCompositePolicy`,
whose only behavior was the same threshold predicate already supplied by
Moirai's `AdaptiveWithThreshold<N>`. The wrapper duplicated the execution
policy vocabulary and kept a public Apollo module with no Apollo-specific
state or invariant.

## Decision

Delete the Apollo policy module and instantiate
`moirai::AdaptiveWithThreshold<RADIX_PARALLEL_CHUNK_THRESHOLD>` directly at
the radix-composite kernel boundary. Keep the threshold in Apollo's tuning
module because it is a workload decision; keep policy semantics in Moirai,
the SSOT for execution strategy.

## Theorem / contract

For every workload length `n`, the removed policy and the selected Moirai type
have the same predicate by construction:

\[
  \operatorname{parallelize}(n) \iff n \ge
  \texttt{RADIX\_PARALLEL\_CHUNK\_THRESHOLD}.
\]

The boundary regression asserts both sides of the predicate. FFT value
semantics remain covered by the existing radix-composite and package suites;
no arithmetic or scheduling implementation changes.

## Consequences

`apollo-fft` advances from 0.24.0 to 0.25.0 because the public policy module
and type are removed. No compatibility export is retained; callers use the
canonical Moirai policy seam.
