# ADR 0018: Partition the Hilbert GPU verification tree

## Status

Accepted for the D8 Hilbert verification slice.

## Context

`apollo-hilbert` keeps 410 lines of private GPU verification in one module. It
mixes capability metadata, invalid-length rejection, analytic-signal and
quadrature CPU differentials, inverse projection, Leto host boundaries,
represented storage, and precision-profile rejection. Those independent
contracts have no private module boundary.

Hephaestus owns generic device acquisition, buffers, dispatch, submission, and
transfer. Hilbert retains frequency-mask semantics, numerical values, and
transform-specific errors; moving tests must not recreate a consumer-side
provider layer.

## Decision

Replace the private monolith with a verification manifest and concern-named
leaves for metadata and rejection, forward analytic/quadrature execution,
inverse projection, Leto host boundaries, typed storage, explicit precision
boundaries, and shared device acquisition. Every moved test retains its
existing fixture, CPU oracle, and finite-precision bound.

## Theorem preservation

For the DFT convention `X[k] = sum_n x[n] exp(-2 pi i k n / N)`, the Hilbert
multiplier is `-i sgn(k)` away from DC and, for even lengths, Nyquist. On the
subspace with zero DC and Nyquist coefficients, applying it twice gives
`(-i sgn(k))^2 = -1`, hence `H(H(x)) = -x` in exact arithmetic. The inverse
GPU mask applies `-H` and therefore reconstructs only that projection. The
inverse leaves retain their existing CPU-differential and finite-precision
checks. This is a proof sketch plus empirical numerical evidence, not a
machine-checked proof.

## Consequences

The private test tree gains bounded, concern-named leaves without changing the
public Hilbert API, frequency-mask semantics, Leto boundary, or Hephaestus
ownership. The manifest contains no test implementation.
