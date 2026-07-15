# ADR 0005: SHT dispatch through Hephaestus

## Status

Accepted

## Context

`apollo-sht` previously constructed WGPU shader modules, bind-group layouts,
pipelines, command encoders, and transfers through `apollo-wgpu-helpers`.
That duplicated the device-provider responsibility already owned by
Hephaestus and let raw provider types cross Apollo's transform boundary.

The SHT has two ordered operations.  For each mode `(l, m)` and grid sample
`j`, the first operation materializes the spherical-harmonic basis.  The
second performs the forward quadrature or inverse synthesis matrix product.
The basis must complete before the matrix operation consumes it.

## Decision

`apollo-sht` declares two typed, zero-sized Hephaestus kernels and records
them in one `CommandStream`:

1. `ShtBasisKernel<P>` writes the basis for a compile-time matrix pass `P`.
2. `ShtMatrixKernel<P>` consumes that basis and writes the requested result.

Hephaestus owns provider acquisition, shader preparation, binding validation,
buffer allocation, command recording, submission, and transfer.  Apollo owns
only SHT parameters, grid metadata, mathematical kernel source, and host
boundary conversion.  Leto remains the array boundary; Mnemosyne backs
temporary host conversion storage and Leto output allocation.

The concrete accelerator representation is `Complex32`.  CPU-domain
coefficients remain `Complex64`.  Inverse execution accepts only components
exactly representable as `f32`; explicit quantization is a named API rather
than a silent narrowing path.

## Theorem and proof sketch

Let `Y_l^m` use Condon--Shortley phase and orthonormal normalization.  For a
grid with Gauss--Legendre nodes `x_j = cos(theta_j)`, corresponding weights
`w_j`, and `N_phi` uniform longitudes, the forward pass evaluates

```text
a_l^m = sum_j sum_q f(theta_j, phi_q) conj(Y_l^m(theta_j, phi_q))
        w_j (2 pi / N_phi).
```

The inverse pass evaluates

```text
f_hat(theta_j, phi_q) = sum_(l=0)^L sum_(m=-l)^l a_l^m Y_l^m(theta_j, phi_q).
```

For functions band-limited to `L`, with a grid satisfying the plan's
Gauss--Legendre and azimuthal sampling constraints, the product quadrature is
exact for these basis products in exact arithmetic.  Orthonormality therefore
gives `a_l^m` as the harmonic coefficient and synthesis reconstructs the
band-limited function.  The implementation preserves the two formulae because
the forward marker selects conjugation and quadrature weighting, while the
inverse marker selects neither; command-stream ordering establishes the
write-before-read dependency from basis generation to matrix reduction.

Finite-precision WGPU execution is not this exact-arithmetic theorem.  It has
empirical differential evidence against the CPU `Complex64` implementation
with a tolerance derived from the direct summation depth.  Exact CPU values
that cannot enter `Complex32` are rejected before provider allocation or dispatch.

## Consequences

- Apollo contains no raw `wgpu`, `pollster`, or helper dispatch API in the SHT
  provider path.
- The source is partitioned by basis and matrix responsibilities, preserving
  one generic direction-parameterized implementation instead of duplicated
  forward and inverse orchestration.
- Callers requiring an intentional `Complex64 -> Complex32` approximation use
  the explicit quantization boundary.
