# Apollo GFT

`apollo-gft` owns graph Fourier transforms over real weighted undirected
graphs.

## Architecture

```text
src/
  domain/          graph adjacency validation and error contracts
  application/     reusable graph Fourier plan
  infrastructure/  Laplacian and eigensystem construction
  verification/    spectral, roundtrip, and property tests
```

`GftPlan` builds the combinatorial Laplacian and stores the orthonormal
eigenvector basis as the transform matrix.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` storage and owner `f64` graph-basis multiply.
- `LOW_PRECISION_F32`: `f32` storage converted through the owner path and
  quantized once into the caller-owned output.
- `MIXED_PRECISION_F16_F32`: `f16` storage converted through the owner path and
  quantized once into the caller-owned output.

Profile/storage mismatches return `GftError::PrecisionMismatch`.

## Accelerator Contract

The optional WGPU boundary retains the same graph-basis formula but fixes its
arithmetic contract at `f32`: native `f32` storage dispatches without host
conversion and `f16` storage is promoted once before dispatch then quantized
once after transfer. `f64` is deliberately excluded from this accelerator API;
accepting it would conceal a precision-narrowing boundary.

Apollo provides the WGSL source, graph order, direction, and basis layout.
Hephaestus owns device acquisition, typed buffers, pipeline preparation,
binding validation, command streams, dispatch, synchronization, and transfer.
Leto remains the host-array boundary; contiguous views are borrowed and
strided views materialize only their logical host order before upload.

## Mathematical Contract

For symmetric adjacency `A`, degree matrix `D`, and Laplacian `L = D - A`, the
real-symmetric eigendecomposition gives `L = U Lambda U^T` with `U^T U = I`.
The graph Fourier transform is

```text
X = U^T x
x = U X
```

so inverse reconstruction follows from orthonormality.

The accelerator evaluates the same indexed sums in `f32`:

```text
X[k] = sum_i U[i + kN] x[i]
x[i] = sum_k U[i + kN] X[k]
```

The executable verification uses the path-four graph. Each output has four
products and three additions, so its first-order dot-product bound is
`gamma_7`; the CPU differential tolerance is `64 * epsilon_f32 = 2^-17`,
covering that rounding plus basis quantization without accepting a
normalization error.

## Verification

Tests cover invalid graph contracts, known two-vertex spectra, zero constant
mode for a path graph, eigenbasis orthonormality, weighted graph roundtrips, and
random graph roundtrips. Typed tests cover `f64`, `f32`, mixed `f16`, inverse
roundtrip, caller-owned parity, and precision/profile mismatch rejection.
The accelerator suite differentially checks both directions against the CPU
theorem implementation, roundtrip reconstruction, Leto contiguous and strided
views, caller-owned outputs, explicit `f16` quantization, and profile rejection.
