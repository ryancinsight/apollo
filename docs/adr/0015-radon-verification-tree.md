# ADR 0015: Partition the Radon GPU verification tree

## Status

Accepted for the D8 Radon verification slice.

## Context

`apollo-radon` keeps 536 lines of private GPU verification in one module. It
mixes capability metadata, CPU differentials for projection and backprojection,
Leto host-boundary checks, represented-storage contracts, shape rejection,
the discrete-adjoint identity, and filtered-backprojection checks. Those are
independent contracts with no corresponding module boundary.

Hephaestus already owns generic device acquisition, buffer ownership, and
ordered device submission. Radon retains its transform-specific geometry,
capability, and error contracts; moving tests must not recreate a consumer-side
provider layer.

## Decision

Replace the private monolith with a verification manifest and concern-named
leaves for metadata, projection, backprojection, Leto host boundaries, typed
storage, filtered backprojection, and shared device acquisition. Each moved
test retains its existing fixture, CPU oracle, and finite-precision bound.

## Theorem preservation

ADR 0007 states the discrete-adjoint theorem
`<R f, p> = <f, R* p>` for the paired forward deposit and backprojection
interpolation weights. It also states filtered backprojection as
`(pi / A) R*(h * p)` for uniform angles and the Ram-Lak impulse response.
The backprojection and filtered leaves retain the existing finite-precision
tests for those identities. This remains a proof sketch plus empirical
CPU-differential and adjoint evidence, not a machine-checked proof.

## Consequences

The private test tree gains bounded, concern-named leaves without changing the
public Radon API, projection mathematics, Leto boundary, or Hephaestus
ownership. The manifest contains no test implementation.
