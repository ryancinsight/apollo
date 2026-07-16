# ADR 0009: NUFFT dispatch through Hephaestus

## Status

Accepted on 2026-07-15.

## Context

The former NUFFT transport owned raw WGPU device access, pipeline caching,
bindings, command encoders, submission, transfer, and a helper-device wrapper.
Those mechanics duplicated the Atlas accelerator provider and prevented the
fast gridding path from composing its spread/interpolate stages with dense FFT
execution through one typed stream.

Apollo owns direct-sum and Kaiser--Bessel NUFFT mathematics, plan validation,
and WGSL algorithm source. Hephaestus owns accelerator resources and execution.
Leto remains the CPU array/view boundary and does not model device storage.

## Decision

`apollo-nufft` uses typed Hephaestus descriptors and `WgpuCommandStream` for
every direct and fast 1D/3D operation. Direct descriptors bind positions,
complex input, complex output, and one POD parameter block. Fast descriptors
bind the canonical seven position/value/deconvolution/grid/output buffers plus
parameters. The ordered stream records spread or load, typed `GpuFft3d`, then
extract or interpolate; stream order is the write-before-read contract.

`NufftGpuBuffers1D` and `NufftGpuBuffers3D` own typed provider storage. Type-2
output capacity is `max(mode_count, sample_capacity)`, so reusable execution
cannot underallocate when samples outnumber modes. The public accelerator
boundary accepts and returns `hephaestus_wgpu::WgpuDevice`; it exposes no raw
device, queue, encoder, or helper wrapper.

## Mathematical contract

For samples `c_j` at positions `x_j` and Fourier modes `k`, Type-1 evaluates

```text
F_k = sum_j c_j exp(-2*pi*i*k*x_j/L).
```

Type-2 uses the positive exponential. Under the complex inner product,
conjugating the Type-1 exponential gives the Type-2 term, hence

```text
<Type1(c), F> = <c, Type2(F)>
```

in exact arithmetic. This is the direct-pair adjoint theorem. Kaiser--Bessel
spreading/interpolation approximates that pair on an oversampled grid. The
1D inverse FFT normalizes by grid length `M`, so Type-2 multiplies its loaded
deconvolution values by `M` before the inverse to preserve its unnormalized
convention. The 3D implementation retains its already normalized convention.

The theorem is a proof sketch of the mathematical contract, not a
machine-checked proof. CPU differential, adjoint, normalization, and
real-device reusable-buffer tests are empirical finite-precision evidence.

## Compatibility and migration

`apollo-nufft` 0.4.0 removes the public `wgpu_backend` forwarding module and
the public verification-only module. Version 0.5.0 removes the unused
`nufft_wgpu_available` boolean probe: it only erased the acquisition failure
from `NufftWgpuBackend::try_default()`. Consumers import GPU plans, errors,
capabilities, buffers, and `NufftWgpuBackend` directly from the
`apollo_nufft` root, then handle the typed provider acquisition result.
Verification remains private test infrastructure and has no runtime
replacement.

## Consequences

The release has no compatibility shim or duplicate transport surface. The
single root export path retains typed Hephaestus ownership, while Leto remains
the host-array boundary. `NufftWgpuBackend::try_default` remains only because
NuFFT must request its transform-specific seven-storage-buffer lower bound
through the provider; it does not implement a device API. `cargo
semver-checks` classifies the removed public paths as breaking pre-1.0 changes;
finite-precision test evidence does not constitute a machine-checked proof.
