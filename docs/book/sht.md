# Spherical Harmonic Transform

The SHT expands functions sampled on a spherical surface in spherical
harmonics — the sphere's analogue of the Fourier basis — for analysis and
synthesis of global fields. Apollo owns the transforms in `apollo-sht`,
built around `ShtPlan`.

```rust
{{#include ../../crates/apollo-sht/examples/book_sht_constant.rs}}
```

## Separated execution

`ShtPlan::new` binds a latitude/longitude grid and a maximum degree,
precomputing the Gauss–Legendre nodes and weights for the latitude axis.
The longitude axis factors exactly: the `e^{imφ}` factor of each harmonic
is a DFT kernel, so the sum over longitudes is one DFT bin of the latitude
row. Forward analysis and inverse synthesis both ride this separation,
with real and complex sample paths over the same plan.

## Orthonormal coefficients

Coefficients live in `SphericalHarmonicCoefficients`, indexed by degree
and signed order through `get`/`set`. The basis is orthonormal, so a
constant `1/sqrt(4π)` field carries exactly one unit coefficient — the
sample checks `Y_00 == 1`, every other harmonic at zero, and the
round-trip back to the grid.
