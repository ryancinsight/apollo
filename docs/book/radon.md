# Radon Transform

The Radon transform integrates an image along parallel-beam lines into a
sinogram, and filtered backprojection inverts it — the mathematics of CT
reconstruction. Apollo owns the discrete projections and the CT-style
adjoint in `apollo-radon`, built around `RadonPlan`.

```rust
{{#include ../../crates/apollo-radon/examples/book_radon_adjoint.rs}}
```

## Geometry-bound plans

`RadonPlan::new` binds the image shape, the projection angles, and the
detector row (count plus spacing) into a `ParallelBeamGeometry`. `forward`
projects an image to a `Sinogram` indexed by angle and detector, while
`backproject` applies the exact adjoint — the sample checks the adjoint
identity `<Ax, Ax> == <x, A'Ax>` rather than a pseudo-round-trip, since
unfiltered backprojection is not an inverse.

## Filtered reconstruction

`filtered_backprojection` pairs ramp filtering (`ramp_filter_projection`)
with the adjoint to reconstruct the image, the discrete CT pipeline.
Caller-owned `forward_into`/`backproject_into` variants and the typed
storage seam cover hot loops and low-precision inputs.
