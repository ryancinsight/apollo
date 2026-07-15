# ADR 0012: DCT/DST GPU verification ownership

## Status

Accepted for the first D8 vertical slice.

## Context

`apollo-dctdst` has a 796-line GPU verification module containing capability,
one-dimensional CPU-differential, mixed-storage, Leto-boundary, dimensional,
and rejection contracts. It mixes unrelated verification concerns, obscuring
the transform theorem each assertion checks.

D8 identifies repeated GPU device/error/capability and verification scaffolding
across transforms as provider-owned work. Apollo must not answer that gap with
another device wrapper or a compatibility layer.

## Decision

Keep DCT/DST-specific mathematical values and CPU oracles in
`apollo-dctdst`, but partition this suite into private verification leaves by
contract. A single local support leaf owns only repeated value-comparison and
device-availability mechanics. It must not own device handles, kernels, or
transform execution.

The later cross-transform extraction remains conditional on a Hephaestus-owned
provider contract. This slice therefore improves the vertical tree without
pre-empting upstream ownership.

## Verification theorem

For a separable DCT-II on an `n x n` or `n x n x n` field, applying the same
one-dimensional operator along every axis equals the tensor-product operator.
The dimensional CPU oracle applies those axis operators independently; the GPU
result is compared componentwise to that independent construction. Inverse
tests additionally check `T^{-1}(T(x)) = x` within the existing derived f32
rounding bounds.

This is a proof sketch. Evidence is value-semantic CPU-differential and
round-trip testing, plus the existing typed provider boundary; it is not a
machine-checked proof or a GPU performance claim.

## Consequences

- The test tree gains bounded, discoverable leaves without duplicating
  transform execution.
- Production GPU code and its Hephaestus ownership remain unchanged.
- A future generic harness must be implemented at the provider boundary, not
  copied into each Apollo transform crate.
