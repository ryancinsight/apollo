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

## Hephaestus accelerator contract

The `wgpu` feature routes f32 dense-FFT execution through
`hephaestus_wgpu::WgpuDevice`. Apollo supplies zero-sized descriptors for the
radix, pack/unpack, and Bluestein stages; Hephaestus owns typed buffers,
pipeline preparation, binding validation, ordered command streams, submission,
and transfer. `GpuFft3d::encode_forward_split` and
`GpuFft3d::encode_inverse_split` accept only provider-typed split-complex
buffers and a provider command stream, so downstream composed operations such
as NUFFT do not acquire a raw device, queue, buffer, or encoder.

For dimensions `N_x`, `N_y`, and `N_z`, the provider stream records forward
axes in Z/Y/X order and inverse axes in X/Y/Z order. The transform convention is

```text
X[k_x, k_y, k_z] = sum_{x,y,z} x[x,y,z]
  exp(-2*pi*i*(k_x*x/N_x + k_y*y/N_y + k_z*z/N_z))
```

with inverse positive exponent and `1/(N_x*N_y*N_z)` normalization. Root-of-
unity orthogonality proves `F^-1(F(x)) = x` in exact arithmetic. This is a
mathematical proof sketch, not a machine-checked proof. The real-device typed
stream tests verify a 2x2x2 delta exactly and a 2x3x2 Bluestein delta within
the documented f32 `gamma_256` rounding bound.

Native f16 shader execution remains a separate migration scope. It is not a
fallback for the provider-native f32 path and retains its own documented
precision contract until it is converted to the same descriptor boundary.

## Verification

Tests cover analytical small transforms, radix-2 and Bluestein parity against
direct DFT, inverse roundtrips, typed external-buffer command-stream
composition, Parseval-style energy checks, linearity,
caller-owned output paths, slice-level real-forward parity and shape rejection,
precision profile behavior, and 2D/3D separable axis execution.
