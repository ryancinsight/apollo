# ADR 0030: CUDA FFT through the Hephaestus provider boundary

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking provider-surface extension

## Context

Apollo's FFT backend contract described CPU and WGPU implementations, while the
backlog retained a CUDA residual. Hephaestus now provides `CudaDevice`, typed
buffers, authored-kernel compilation, binding validation, ordered streams, and
synchronization through `hephaestus-cuda`. A consumer-local driver wrapper or
a second backend-specific FFT descriptor would duplicate that provider role and
allow WGPU/CUDA kernel contracts to drift.

## Decision

Add a CUDA feature to `apollo-fft` and expose `CudaBackend` plus a
one-dimensional `CudaFft1d` plan. The public construction boundary accepts an
already-acquired `hephaestus_cuda::CudaDevice`; Apollo does not acquire CUDA
devices or import CUDA-driver APIs. The plan accepts a Leto `Array1<Complex32>`
and reuses its typed split-complex CUDA buffers, prepared kernel handles, and
host staging on every call. Submission has no extra device-wide synchronization:
the typed device-to-host transfer is the provider's synchronous completion
boundary before output mutation.

Move the dialect-neutral `FftParams`, zero-sized kernel descriptors, and radix
stage values into one transport leaf. WGSL and CUDA C implement their
respective `KernelSource` dialects on those same descriptors. CUDA implements
only the radix-two entries needed for the initial one-dimensional f32 scope;
its capability descriptor truthfully rejects 2D, 3D, real-to-complex, and
mixed-precision planning.

## DFT theorem and numerical bound

For `N = 2^m`, the plan implements the forward convention

`X[k] = sum(j = 0..N-1, x[j] exp(-2 pi i j k / N))`

and the inverse convention

`x[j] = (1 / N) sum(k = 0..N-1, X[k] exp(2 pi i j k / N))`.

Bit reversal places each input at the index formed by reversing its `m` binary
digits. Radix stage `s` combines the two length-`2^s` subtransforms with the
twiddle `exp(+/- 2 pi i r / 2^(s+1))`; induction over the `m` stages yields the
forward or inverse DFT above. In exact arithmetic, substituting the forward
sum into the inverse yields

`(1 / N) sum(k, l, x[l] exp(2 pi i k (j-l) / N)) = x[j]`,

because the root-of-unity sum is `N` when `j = l` and zero otherwise.

The device-present CPU and WGPU differential tests use
`2 gamma_256 ||x||_1`, where
`gamma_256 = 256 u / (1 - 256 u)` and `u = epsilon_f32 / 2`. Each side has a
conservative `gamma_256 ||x||_1` bound for the bounded eight-point regression
kernel, and the triangle inequality gives the factor two. The inverse test
uses `gamma_256 (1 + ||x||_1)` after the normalized inverse. These are
finite-precision empirical checks, not a machine-checked proof of CUDA
transcendental implementations.

## Consequences

- Hephaestus remains the sole owner of CUDA acquisition, driver interaction,
  buffers, compilation, launch mechanics, and synchronization.
- Apollo owns the FFT equation, its typed kernel interface, plan validation,
  and Leto host boundary once.
- `BackendKind` gains `Cuda`; downstream exhaustive matches must handle the
  new provider family.
- CUDA tests skip only `HephaestusError::AdapterUnavailable`; every other
  provider failure remains visible.
- On this Windows GNU validation host, the installed driver supplies
  `nvcuda.dll` but not its GNU import archive. The checked test command adds a
  generated `libcuda.dll.a` from that installed driver in the ignored shared
  target tree. This is linker environment setup only; it adds no source,
  manifest, or runtime fallback to Apollo.
