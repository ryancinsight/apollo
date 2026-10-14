# Fractional Fourier Transform

The FrFT rotates a signal by an arbitrary angle in the time-frequency
plane, generalizing the Fourier transform (a quarter turn) to a continuous
family with the identity and reversal at its endpoints. Apollo owns the
reference implementation in `apollo-frft`, built around `FrftPlan`.

```rust
{{#include ../../crates/apollo-frft/examples/book_frft_orders.rs}}
```

## Orders modulo four

`FrftPlan::new` takes a length and a fractional order, reduced modulo
four: order zero is the identity, order one the Fourier transform, with
integer rotations on an exact fast path. `inverse` is the forward FrFT of
negated order, so forward-then-inverse round-trips at every angle. A free
`frft` function covers one-shot evaluations.

## Unitary variant

`UnitaryFrftPlan` implements the Candan–Grünbaum eigendecomposition-based
discrete FrFT over the `GrunbaumBasis` eigenvectors, for work that needs
the strictly unitary discrete form rather than the sampled continuous
kernel.
