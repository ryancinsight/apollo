# Apollo FFT

`apollo-fft` owns Apollo's dense CPU Fourier transform implementation, shared
shape contracts, backend abstractions, and cache-backed plan surfaces.

## Architecture

```text
src/
  domain/          backend, error, and shape contracts
  application/     FFT plans and plan cache orchestration
  infrastructure/  CPU backend transport
```

The dense FFT crate is the single source of truth for 1D, 2D, and 3D uniform
FFT plans. NUFFT and SFT logic live in their own crates.

## Mathematical Contract

The forward complex FFT computes

```text
X[k] = sum_n x[n] exp(-2*pi*i*k*n/N)
```

The inverse computes the conjugate-sign transform and applies Apollo's selected
normalization. The kernel strategy auto-selects radix-2 Cooley-Tukey for
power-of-two lengths and Bluestein chirp-Z for arbitrary lengths. The direct
DFT kernel remains a crate-local reference for verification.

2D and 3D plans execute separable axis passes. Contiguous row/depth-axis passes
operate directly on backing-slice chunks through Moirai, avoiding full-field
lane-copy vectors and scatter copies. Non-contiguous axes still gather one lane
buffer per lane before scattering because strided Leto views are not contiguous
in memory.

The 1D real-forward plan surface supports Leto-owned allocation and caller-owned
slice output paths. Slice execution lets downstream crates reuse existing real
input slices while still sharing the same real FFT owner kernel. The typed plan
surface supports `f64` storage with `Complex64` compute, `f32` storage with
`Complex32` compute, and mixed `f16` storage with `f32` compute. The 3D typed
`*_into` paths accept caller-owned output and scratch buffers for all three
precision profiles to avoid repeated spectrum allocation in memory-bound
workloads.

## Verification

Tests cover analytical small transforms, radix-2 and Bluestein parity against
direct DFT, inverse roundtrips, Parseval-style energy checks, linearity,
caller-owned output paths, slice-level real-forward parity and shape rejection,
precision profile behavior, and 2D/3D separable axis execution.
