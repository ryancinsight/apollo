# ADR 0010: Shared Leto interop ownership

## Status

Accepted on 2026-07-15.

## Context

`apollo-fft` is an FFT implementation crate. Its
`application::utilities::leto_interop` module was also the workspace source of
truth for Leto view materialization and Mnemosyne-backed output construction.
Seventeen transform crates reached through that FFT-private namespace, and
their local forwarding functions duplicated the same API shape. The dependency
direction therefore made unrelated transforms depend on FFT implementation
structure merely to cross the Leto host-array boundary.

## Decision

`apollo-leto-interop` owns the narrow, rank-polymorphic Leto boundary:

- `view_cow` borrows a C-contiguous view and materializes any other view once
  in Leto logical row-major order;
- `try_dense_from_array` and `try_dense_from_view` construct dense
  Mnemosyne-backed outputs from arbitrary rank-polymorphic Leto arrays/views;
  `try_dense_from_slice` is the corresponding computed-slice boundary; and
  one-dimensional constructors retain their move/copy distinction.

Every transform calls that crate directly. `apollo-fft` deletes its former
utility module, and transform-local forwarding wrappers are deleted in the
same migration. Precision storage/compute comparison is not Leto interop; it
becomes a method on `apollo_fft::PrecisionProfile`, its owning domain type.

## Representation theorem

For any Leto view `V` with logical row-major element sequence `v_i`, the
returned `C = view_cow(V)` satisfies `C[i] = v_i` for every valid logical
index. If `V.as_slice()` succeeds, `C` is borrowed and no element copy occurs.
Otherwise Leto's `ElementIter` visits precisely that logical row-major
sequence, so collecting it produces the same values in owned contiguous
storage.

For a dense output constructor given source shape `s`, the returned array has
shape `s` and the same logical element sequence. The contiguous branch copies
the source slice; the strided branch collects that sequence once before Leto
checks cardinality. These are proof sketches over Leto's documented slice and
iterator contracts, not machine-checked proofs. Rank-one strided, rank-two
strided/transposed, and rank-three output cases are value-semantic regression
evidence.

## Consequences

The crate graph becomes `transform -> apollo-leto-interop -> leto`, rather
than `transform -> apollo-fft utility`. The boundary has one generic
implementation; no compatibility re-export or local adapter remains. This is
an architectural, pre-1.0 breaking change for the previously public FFT
utility path.
