# ADR 0032: Direct Leto output construction

## Status

Accepted on 2026-07-16.

## Context

`apollo-leto-interop` is the SSOT for rank-polymorphic view materialization
and computed-slice output construction. Its `try_dense_from_contiguous`
function duplicated the provider call already available at every contiguous
FFT output boundary. The function was used only by the four two- and
three-dimensional real FFT API paths, while the accepted D7 contract names
`try_dense_from_array`, `try_dense_from_view`, and `try_dense_from_slice` as
the interop surface.

## Decision

Delete `try_dense_from_contiguous`. The FFT API constructs its Mnemosyne-backed
result directly with
`leto::Array::from_mnemosyne_slice(output.shape(), output.as_slice())` after
the owning FFT kernel returns a contiguous output. No forwarding function,
alias, or compatibility path remains.

## Output-shape theorem

Let `O` be a contiguous Leto FFT output with shape `s` and logical sequence
`o_i`. The direct construction returns a Mnemosyne-backed array `M` with
shape `s` and `M[i] = o_i` for every valid index. `O.as_slice()` exposes the
same logical order for a contiguous Leto array, and
`from_mnemosyne_slice(s, ...)` preserves that sequence after validating the
shape/cardinality invariant. This is a provider-contract proof sketch, not a
machine-checked proof; two- and three-dimensional forward/inverse parity tests
exercise the boundary against the existing array API.

## Consequences

The interop crate has one dense-slice constructor instead of a second
contiguous-array forwarding shape. Apollo FFT remains the transform owner,
Leto remains the host-array owner, and Mnemosyne remains the returned-storage
owner. Removing the public helper is a pre-1.0 breaking change for the
`apollo-leto-interop` package; all in-repository callers migrate in this
increment.
