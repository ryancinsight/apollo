# Chirp Z-Transform

The CZT evaluates the z-transform along a chirp contour — a spiral of
points `A · W^{-k}` — so it zooms spectra to arbitrary frequency bands at
arbitrary resolution instead of the DFT's fixed grid. Apollo owns the
Bluestein-plan implementation in `apollo-czt`, built around `CztPlan`.

```rust
{{#include ../../crates/apollo-czt/examples/book_czt_delta.rs}}
```

## Bluestein execution

`CztPlan::new` takes the input length, output count, start `a`, and step
`w`, then precomputes the chirp factors and an internal FFT plan over the
padded convolution length. `forward` runs the `O(P log P)` Bluestein
convolution, while `forward_direct` keeps the definitional double loop for
small problems and cross-checks. The delta sample holds for any contour:
with `a = 1` every output point sees exactly the first input sample.

## Inverse support

`inverse` recovers the input through a Björck–Pereyra Vandermonde solve
in `O(N²)` time. Inversion needs a square problem — a rectangular `M ≠ N`
CZT is not uniquely invertible — and returns `NotInvertible` when two
evaluation points coincide. Free functions (`czt`, `czt_direct`, and the
Leto-view twins) cover one-shot evaluations without plan amortization.
