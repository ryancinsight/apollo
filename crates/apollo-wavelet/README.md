# Apollo Wavelet

`apollo-wavelet` owns discrete and continuous wavelet transforms for Apollo
multiresolution analysis.

## Architecture

```text
src/
  domain/          wavelet descriptors, errors, and coefficient storage
  application/     reusable DWT and CWT plans
  infrastructure/  orthogonal filter banks and CWT mother-wavelet kernels
  verification/    analytical, boundary, and property tests
```

`DwtPlan` and `CwtPlan` are the authoritative execution surfaces. Domain types
define admissible wavelets and coefficient ownership; infrastructure kernels
only evaluate mathematical primitives.

Typed execution uses the same owner kernels for `f64`, `f32`, and mixed `f16`
storage. The typed DWT and CWT APIs convert represented input into the
authoritative `f64` arithmetic path and quantize once into caller-owned output
buffers, so storage precision does not create parallel algorithm families.

The Haar GPU path is separate: Leto owns host views and Mnemosyne-backed
outputs, while Hephaestus owns device buffers, parameter upload, binding,
dispatch, ordered prefix copies, and transfer. Its concrete `f32` contract
admits native `f32` and explicit `f16` storage only; `f64` is rejected at
compile time rather than silently narrowed.

## Mathematical Contract

The DWT uses orthogonal analysis/synthesis filters with periodic boundaries. For
Haar and Daubechies-4, the synthesis filters are the quadrature mirror inverse
of the analysis filters, so multilevel inverse reconstruction recovers the
original power-of-two signal up to floating-point roundoff.

For the GPU Haar kernel, one pass maps `(a, b)` to
`((a + b) / sqrt(2), (a - b) / sqrt(2))`; its transpose is synthesis.
Consequently each pass is orthonormal, preserves squared Euclidean energy, and
the reverse-level synthesis composition is the inverse of the forward
composition. This is a proof sketch; real-device CPU differential, Parseval,
and roundtrip tests provide empirical evidence for the implemented `f32` path.

The CWT computes

```text
W_x(a, b) = a^(-1/2) sum_n x[n] psi((n - b) / a)
```

for positive scale `a`. Ricker and DC-corrected real Morlet wavelets are
zero-mean analysis kernels, so constant signals have no continuous-limit
wavelet response.

## Verification

The crate verifies analytical Haar coefficients, Haar/Daubechies-4 inverse
reconstruction, invalid contracts, CWT impulse localization, zero-signal
response, Morlet finite coefficients, Morlet zero-mean admissibility, typed
DWT/CWT parity for `f64`, represented-input parity for `f32` and mixed `f16`,
inverse DWT roundtrip for `f32`, shape rejection, and profile/storage mismatch
rejection.

The Hephaestus GPU suite also verifies analytical two-sample Haar values,
forward CPU differential, inverse reconstruction, Parseval conservation,
Leto contiguous and strided boundaries, typed `f16` quantization, and the
compile-fail exclusion of `f64` GPU storage.
