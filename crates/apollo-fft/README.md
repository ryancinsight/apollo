# Apollo FFT

`apollo-fft` owns Apollo's dense CPU Fourier transform implementation, shared
shape and precision contracts, plan caches, and the generic WGPU transform
scaffold used by sibling transform crates.

## Architecture

```text
src/
  domain/          backend, error, precision, and shape contracts
  application/     CPU FFT plans, kernels, and cache orchestration
  infrastructure/  CPU/CUDA transport and generic transform infrastructure
```

The crate is the single source of truth for Apollo's one-, two-, and
three-dimensional CPU FFT plans. NUFFT and sparse Fourier logic live in their
own crates. Dense WGPU FFT algorithms and resources live in Hephaestus.

## Mathematical contract

The forward complex FFT computes

```text
X[k] = sum_n x[n] exp(-2*pi*i*k*n/N)
```

The inverse uses the positive exponent and applies Apollo's selected
normalization. The CPU strategy selects radix, mixed-radix, Rader, or
Bluestein construction from the validated shape. Direct DFT kernels remain
reference surfaces for verification rather than production fallbacks.

Two- and three-dimensional plans execute separable axis passes. C-dense Leto
views, including offset views, operate on their backing block. Fortran-dense
and general strided views assign once into reusable logical C-order staging,
transform, and assign back; warmed staging allocates nothing. Row and
depth-axis passes operate on chunks through Moirai, while non-contiguous axes
transpose through Leto Ops into plan-owned scratch. Leto selects exact Hermes
register tiles for supported high-count small-matrix batches and retains its
generic allocation-free assignment elsewhere.

The typed CPU plan surface supports f64 storage/compute, f32 storage/compute,
and mixed f16 storage with f32 compute. Caller-owned output and scratch paths
avoid repeated result and workspace allocation.

## Accelerator boundary

Use `hephaestus_wgpu::WgpuFftOps` for dense WGPU FFT execution. Consumers pair
typed split-complex device buffers with a Leto `Layout<R>` through
`hephaestus_core::FftOperands`, then retain the returned prepared plan. Apollo
does not expose a WGPU FFT facade, shaders, staging workspace, or native-f16
feature.

The `wgpu` feature remains because sibling Apollo transform crates share the
generic `WgpuTransformBackend`, plan, storage, and error contracts from this
crate. Those contracts do not implement dense FFT arithmetic. The independent
CUDA plan remains governed by ADR 0030.

## Verification

Tests cover analytical transforms, direct-DFT differential equality within
derived finite-precision bounds, inverse roundtrips, Parseval-style identities,
linearity, caller-owned output paths, scalar instantiations, Leto layout
preservation, and separable two- and three-dimensional execution. Accelerator
FFT conformance and rank coverage are verified in Hephaestus; Apollo validation
adds a direct Hephaestus-versus-Apollo CPU differential.
