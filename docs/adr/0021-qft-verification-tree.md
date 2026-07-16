# ADR 0021: QFT GPU verification tree

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking verification-boundary cleanup

## Context

`apollo-qft` keeps static metadata, CPU differentials, inverse reconstruction,
Leto host-boundary, represented-storage, precision rejection, and pre-dispatch
rejection contracts in one 381-line GPU verification module. The module is a
public transport path even though its contents exist only under `cfg(test)`.

## Decision

Replace the monolith with a test-only manifest and concern-named leaves.
`support` owns only device acquisition and the established finite-precision
bounds; it does not wrap or implement a provider API. Hephaestus retains device
mechanics; Apollo retains QFT mathematics; Leto owns host-array/view boundaries.

For the QFT matrix `U`, `U[k, j] = exp(2 pi i k j / N) / sqrt(N)`. Discrete
Fourier orthogonality gives
`(U^dagger U)[j, j'] = (1 / N) sum_k exp(2 pi i (j - j') k / N) = delta[j, j']`.
Thus `U^dagger U = I`, so inverse execution reconstructs the input and forward
execution preserves the L2 norm. The CPU plan provides the independent oracle.

The public test-only transport verification path is removed rather than kept as
an empty release wrapper. The package version is selected only after
`cargo-semver-checks` reports the affected public paths.

## Consequences

- Static and device-present contracts run independently without changing QFT
  execution, fixture values, or provider fallback behavior.
- The theorem is a proof sketch; finite-precision tests provide empirical
  evidence, not a machine-checked proof.
- No shared Apollo device abstraction is introduced; cross-transform provider
  consolidation remains Hephaestus-owned.
