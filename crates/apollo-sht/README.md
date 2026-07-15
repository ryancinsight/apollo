# Apollo SHT

`apollo-sht` owns spherical harmonic transforms for functions sampled on a
spherical surface.

## Architecture

```text
src/
  domain/          spherical grid metadata, errors, and coefficient storage
  application/     reusable SHT plan
  infrastructure/  Gauss-Legendre and spherical harmonic kernels
  verification/    quadrature, basis, and roundtrip tests
```

`ShtPlan` owns sampling metadata and transform degree bounds. Infrastructure
owns associated Legendre evaluation, normalization, and quadrature weights.

Typed execution supports real `f64`, `f32`, and mixed `f16` sample storage plus
complex `Complex64`, `Complex32`, and mixed `[f16; 2]` coefficient/sample
storage. The Gauss-Legendre quadrature, spherical harmonic basis, and synthesis
remain the authoritative `f64`/`Complex64` owner path; typed APIs convert at
the boundary and write into caller-owned arrays.

With the `wgpu` feature, `ShtWgpuBackend` is a Hephaestus client, not a second
WGPU provider. Apollo supplies typed basis and matrix kernel descriptions;
Hephaestus owns device acquisition, allocation, shader preparation, binding
validation, ordered command-stream submission, and transfer. Leto remains the
host-array boundary and Mnemosyne supplies temporary conversion storage and
Leto output allocation. The concrete GPU representation is `Complex32`.
CPU-domain `Complex64` coefficients enter inverse GPU execution only when each
component is exactly representable; callers that choose approximation must use
the explicit `quantize_coefficients` boundary.

## Mathematical Contract

Complex spherical harmonics are

```text
Y_l^m(theta, phi) = N_lm P_l^m(cos(theta)) exp(i m phi)
```

with Condon-Shortley phase and orthonormal normalization. Gauss-Legendre
quadrature integrates the polar dimension and uniform azimuthal sampling
integrates Fourier modes.

For Gauss-Legendre nodes `x_j = cos(theta_j)`, weights `w_j`, and `N_phi`
uniform longitudes, the forward GPU pass evaluates

```text
a_l^m = sum_j sum_q f(theta_j, phi_q) conj(Y_l^m(theta_j, phi_q))
        w_j (2 pi / N_phi).
```

Its inverse evaluates `f_hat(theta_j, phi_q) = sum_lm a_l^m Y_l^m`. For a
function band-limited to the plan degree and a grid satisfying the plan's
sampling constraints, product quadrature and spherical-harmonic orthonormality
recover the coefficient and synthesize the function in exact arithmetic. The
implementation encodes the two formulas with a compile-time direction marker:
forward basis generation conjugates and weights; inverse basis generation does
neither. Hephaestus command-stream ordering makes basis completion precede the
matrix pass. Finite-precision GPU agreement with the CPU `Complex64` oracle is
empirical differential evidence, not this exact-arithmetic theorem.

## Verification

Tests cover known associated Legendre values, Gauss-Legendre weight sums and
polynomial exactness, invalid sampling, constant-surface coefficients,
single-mode reconstruction, shape mismatches, small-degree roundtrip, typed
real and complex forward parity, typed inverse roundtrip, mixed `f16`
coefficient parity, profile/storage mismatch rejection, direct GPU versus CPU
quadrature/synthesis, and rejection of non-representable inverse coefficients
before provider allocation or dispatch.
