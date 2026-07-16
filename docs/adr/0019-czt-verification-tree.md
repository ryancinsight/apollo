# ADR 0019: CZT GPU verification tree

- Status: Accepted
- Date: 2026-07-16
- Change class: architecture, private verification structure

## Context

`apollo-czt` kept static metadata, direct CPU differential, impulse, Leto
host-boundary, represented-storage, precision rejection, pre-dispatch rejection,
and inverse contracts in one private 395-line GPU verification module. The
module crossed the concern boundary used by the other D8 verification trees.

## Decision

Replace the monolith with a test-only manifest and concern-named leaves. The
`support` leaf owns only device acquisition, shared fixtures, and the existing
finite-precision bounds; it does not wrap or implement a provider API.
Hephaestus remains the sole owner of GPU acquisition and execution mechanics.

For an input `x_n`, the CZT contract is

`X_k = sum_(n=0)^(N-1) x_n A^(-n) W^(nk)`.

With `A = 1`, `W = exp(-2 pi i / N)`, and `M = N`, this is the forward DFT.
The inverse round-trip test therefore uses that specialization, not arbitrary
CZT parameters. The test keeps the established `8192 eps_f32` bound for two
direct transforms. The direct CPU differential keeps `4096 eps_f32`, derived
from four terms, two polar reconstructions, and 512 epsilon per elementary
operation.

## Consequences

- Static and device-present contracts run independently without changing CZT
  execution, public API, fixture values, or provider fallback behavior.
- The theorem statement is a proof sketch; the tests provide finite-precision
  empirical evidence, not a machine-checked proof.
- No shared Apollo device abstraction is introduced; cross-transform provider
  consolidation remains Hephaestus-owned.
