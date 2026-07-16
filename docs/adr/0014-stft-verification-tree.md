# ADR 0014: Partition the STFT GPU verification tree

## Status

Accepted for the D8 STFT verification slice.

## Context

`apollo-stft` keeps 961 lines of private GPU verification in one module. It
mixes metadata and rejection contracts, CPU differentials, overlap-add
reconstruction, typed/Leto boundaries, non-power-of-two coverage, and
reusable-buffer execution. The tree does not expose those independent
contracts at their natural test-module boundary.

Hephaestus already owns generic device acquisition, limits, and provider
errors. STFT capability and error values remain transform-local because they
state STFT execution support and STFT plan failures; this restructuring does
not add an Apollo provider abstraction.

## Decision

Replace the single private module with a verification manifest and leaves for
metadata/rejections, forward CPU differentials, inverse/overlap-add
reconstruction, typed and Leto host boundaries, and non-power-of-two/reusable
storage contracts. Shared device acquisition stays in one private support
leaf. Every existing test keeps its input, oracle, and derived tolerance.

## Theorem preservation

ADR 0008 records the weighted-overlap-add identity
`sum_m x[t] w[t-mH]^2 / sum_m w[t-mH]^2 = x[t]` where its denominator is
non-zero. The inverse/reconstruction leaf preserves that theorem's
finite-precision tests unchanged. This is a proof sketch plus empirical
CPU-differential and reconstruction evidence, not a machine-checked proof.

## Consequences

The private tree gains bounded, concern-named leaves without changing the
public STFT API, GPU dispatch, Leto boundary, or Hephaestus ownership. The
module manifest contains no test implementation.
