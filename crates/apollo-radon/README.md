# Apollo Radon

`apollo-radon` owns discrete parallel-beam Radon projections and CT-style
filtered backprojection.

## Architecture

```text
src/
  domain/          geometry contracts, errors, and sinogram storage
  application/     reusable Radon plan
  infrastructure/  forward, adjoint, and ramp-filter kernels
  verification/    projection, adjoint, and filter tests
```

`RadonPlan` owns validated geometry and delegates computation to crate-local
forward, adjoint, and filtering kernels.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` image/sinogram storage with owner `f64`
  projection and adjoint kernels.
- `LOW_PRECISION_F32`: `f32` image/sinogram storage converted through the owner
  path and quantized once into caller-owned outputs.
- `MIXED_PRECISION_F16_F32`: `f16` image/sinogram storage converted through the
  owner path and quantized once into caller-owned outputs.

Profile/storage mismatches return `RadonError::PrecisionMismatch`.

## Mathematical Contract

The forward model treats each pixel value as a point mass at the pixel center
and deposits it onto detector bins with linear weights. Backprojection applies
the same weights in reverse, giving the discrete adjoint identity:

```text
<R f, p> = <f, R* p>
```

Filtered backprojection applies a ramp filter to each projection before the
adjoint step.

The accelerator path uses three typed Hephaestus descriptors. The filtered
sinogram write is encoded before the adjoint read in one command stream, so the
provider's inter-pass ordering establishes that dependency. Leto remains the
CPU array boundary; Hephaestus owns typed buffers, pipeline preparation,
binding validation, command encoding, submission, and readback.

For a uniform angular quadrature, filtered backprojection evaluates

```text
f_hat = (pi / A) R* (h * p),
```

where `A` is the angle count, `R*` is the discrete adjoint, and `h` is the
Ram-Lak impulse response. The adjoint identity is exact for the declared
discrete interpolation pair in exact arithmetic. The filtered reconstruction
is an approximation to the continuous inverse; real-device CPU differential
tests are empirical finite-precision evidence, not a proof of continuous CT
inversion.

## Verification

Tests cover axis-aligned row/column sums, adjoint identity, detector mass
conservation, ramp-filter DC removal, caller-owned filter parity, and invalid
shape/geometry contracts. Typed tests cover `f64`, `f32`, mixed `f16`,
caller-owned forward/backprojection parity, represented-input projection
parity, and precision/profile mismatch rejection.
