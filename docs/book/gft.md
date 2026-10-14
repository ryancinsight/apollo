# Graph Fourier Transform

The GFT expands graph signals in the eigenbasis of the graph Laplacian,
so smooth signals concentrate in the low graph frequencies exactly as
smooth time signals concentrate in low Fourier frequencies. Apollo owns
the transforms over real weighted undirected graphs in `apollo-gft`,
built around `GftPlan`.

```rust
{{#include ../../crates/apollo-gft/examples/book_gft_roundtrip.rs}}
```

## Spectral basis

`GftPlan::from_graph` validates an undirected weighted adjacency matrix
and diagonalizes its Laplacian once. `eigenvalues` exposes the graph
frequencies and `forward` projects a vertex signal onto the eigenbasis;
`inverse` reconstructs the signal from its spectrum. Because the basis is
orthonormal, forward-then-inverse is the identity to rounding, as the
sample checks on a three-node path graph.

## Plan shapes

`from_adjacency` accepts a borrowed matrix view for one-shot plans, while
`from_graph` takes a validated `GraphAdjacency` that can be reused. Both
`forward_into` and `inverse_into` write into caller-owned `leto::Array1`
storage for hot loops, and the typed variants carry the same
storage-profile seam as the other Apollo plans.
