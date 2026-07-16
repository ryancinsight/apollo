# ADR 0016: Partition the Wavelet GPU verification tree

## Status

Accepted for the D8 Wavelet verification slice.

## Context

`apollo-wavelet` keeps 431 lines of private GPU verification in one module. It
mixes capability metadata, invalid-plan rejection, analytical Haar values,
inverse reconstruction, Parseval conservation, CPU differentials, Leto host
boundaries, and represented-storage contracts. Those independent contracts
have no private module boundary.

Hephaestus owns generic device acquisition, buffers, dispatch, and transfer.
Wavelet retains its Haar plan, numerical values, and transform-specific errors;
the test restructuring must not introduce an Apollo-owned provider layer.

## Decision

Replace the private monolith with a verification manifest and concern-named
leaves for metadata, forward execution, inverse laws, Leto host boundaries,
typed storage, and shared device acquisition. Every moved test retains its
existing fixture, CPU oracle, and finite-precision bound.

## Theorem preservation

For one Haar pair, the forward matrix is
`H = (1 / sqrt(2)) [[1, 1], [1, -1]]`. Hence `H^T H = I`; every multilevel
transform is a composition of orthonormal pair passes. Therefore inverse
reconstruction recovers the input and Parseval conservation gives
`sum_i x_i^2 = sum_i (H x)_i^2` in exact arithmetic. The inverse and Parseval
leaves retain the existing finite-precision checks. This is a proof sketch plus
empirical CPU-differential and property evidence, not a machine-checked proof.

## Consequences

The private test tree gains bounded, concern-named leaves without changing the
public Wavelet API, Haar arithmetic, Leto boundary, or Hephaestus ownership.
The manifest contains no test implementation.
