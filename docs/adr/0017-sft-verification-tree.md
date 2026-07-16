# ADR 0017: Partition the SFT GPU verification tree

## Status

Accepted for the D8 SFT verification slice.

## Context

`apollo-sft` keeps 416 lines of private GPU verification in one module. It
mixes capability metadata, plan and input rejection, dense CPU differentials,
inverse sparse reconstruction, Leto host boundaries, represented-storage
contracts, and explicit quantization rejection. Those independent contracts
have no private module boundary.

Hephaestus owns generic device acquisition, buffers, dispatch, submission, and
transfer. SFT retains sparse-domain selection, numerical values, and
transform-specific errors; moving tests must not recreate a consumer-side
provider layer.

## Decision

Replace the private monolith with a verification manifest and concern-named
leaves for metadata and rejection, forward execution, inverse reconstruction,
Leto host boundaries, typed storage, explicit precision boundaries, and shared
device acquisition. Every moved test retains its existing fixture, CPU oracle,
and finite-precision bound.

## Theorem preservation

For the dense convention
`X[k] = sum_n x[n] exp(-2 pi i k n / N)`, the normalized inverse and
`sum_k exp(2 pi i k(n - m) / N) = N delta_nm` give
`x_hat[m] = x[m]` in exact arithmetic. Top-`k` selection is a projection, so
the inverse theorem applies only to the retained sparse spectrum, not discarded
bins. The inverse leaf retains the existing CPU differential and finite-
precision checks. This is a proof sketch plus empirical numerical evidence,
not a machine-checked proof.

## Consequences

The private test tree gains bounded, concern-named leaves without changing the
public SFT API, sparse selection, Leto boundary, or Hephaestus ownership. The
manifest contains no test implementation.
