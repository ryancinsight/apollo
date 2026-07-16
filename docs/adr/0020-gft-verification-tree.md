# ADR 0020: GFT GPU verification tree

- Status: Accepted
- Date: 2026-07-16
- Change class: architecture, private verification structure

## Context

`apollo-gft` held static metadata, CPU differentials, reconstruction, caller-
owned output, Leto host boundaries, represented storage, and precision rejection
contracts in one private 381-line GPU verification module.

## Decision

Replace the monolith with a test-only manifest and concern-named leaves.
`support` owns only device acquisition, the path-four fixture, and the existing
derived tolerance. It does not wrap or implement a provider API. Hephaestus
retains device mechanics; Apollo retains graph-transform mathematics; Leto owns
host-array/view boundaries.

For a real symmetric graph Laplacian, let `U` be the orthonormal eigenbasis.
The graph Fourier pair is `x_hat = U^T x` and `x = U x_hat`. Since `U^T U = I`,
the inverse reconstructs the original signal. The path-four CPU plan supplies
the independent oracle. The existing bound `2^-17` covers first-order dot-product
error, f64-to-f32 basis quantization, and the two launches in the round-trip.

## Consequences

- Static and device-present contracts run independently without changing
  transform execution, fixture values, or provider fallback behavior.
- The prior public test-only verification module and
  `wgpu_backend::verification` path are removed in `apollo-gft` 0.5.0 rather
  than retained as a compatibility shell; runtime callers use the typed
  accelerator API directly.
- The theorem is a proof sketch; the tests are finite-precision empirical
  evidence, not a machine-checked proof.
- No shared Apollo device abstraction is introduced; cross-transform provider
  consolidation remains Hephaestus-owned.
