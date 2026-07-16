# ADR 0022: SHT GPU verification tree

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking verification-boundary cleanup

## Context

`apollo-sht` keeps static metadata, device rejection, CPU differentials, Leto
host-boundary, and typed-storage contracts in one 343-line GPU verification
module. The transport verification module is public even though its contents
exist only under `cfg(test)`.

## Decision

Replace the monolith with a test-only manifest and concern-named leaves.
`support` owns test device acquisition, shared fixture construction, CPU-oracle
representation conversion, and the established finite-precision comparison; it
does not wrap or implement a provider API. Hephaestus retains device mechanics;
Apollo retains SHT mathematics; Leto owns host-array and view boundaries.

The SHT kernel documentation proves the applicable exact-arithmetic contract.
For a field band-limited to degree `L`, `L < N_lat`, and
`2L + 1 <= N_lon`, the longitude sum has discrete Fourier orthogonality and
the `N_lat` Gauss-Legendre rule is exact for the degree-`2L` polar integrand.
Consequently the documented spherical-harmonic basis is orthonormal on this
grid and `inverse(forward(f)) = f`. The CPU plan remains an independent oracle
for the concrete device representation. The existing `2.0e-5` CPU-differential
limit is retained; the theorem is a proof sketch and the finite-precision tests
are empirical evidence, not a machine-checked proof.

The public test-only transport verification path is removed rather than
retained as an empty release wrapper. `cargo-semver-checks` determines the
resulting pre-1.0 version classification.

## Consequences

- Static and device-present contracts run independently without changing SHT
  execution, fixtures, or provider fallback behavior.
- No cross-transform device abstraction is introduced; provider consolidation
  remains Hephaestus-owned.
