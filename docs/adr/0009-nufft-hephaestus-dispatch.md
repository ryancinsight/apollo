# ADR 0009: NUFFT dispatch through Hephaestus

## Status

Accepted on 2026-07-15. Revised on 2026-08-28 under
`APOLLO-FFT-HEPHAESTUS-CUTOVER-2026-08-28`.

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

`apollo-nufft` uses typed Hephaestus descriptors for every direct and fast
1D/3D operation. Direct operations use the flat command stream because their
public per-call contract constructs transient inputs and outputs. Reusable fast
operations bind the canonical seven position/value/deconvolution/grid/output
buffers, a retained parameter buffer, and fixed dispatch grids during
workspace construction. Each call updates the parameter buffer and records
spread or load, a prepared rank-generic Hephaestus FFT, then extract or
interpolate in one grouped sequence; stream order is the write-before-read
contract.

`NufftGpuBuffers1D` and `NufftGpuBuffers3D` own typed provider storage,
including Type-2 coefficient buffers, plus forward and inverse
`WgpuPreparedFft` plans bound to their oversampled grids. Reusable execution
requires an exclusive mutable buffer borrow, so safe callers cannot interleave
host writes or dispatches against one workspace. The buffers also retain host
position and deconvolution conversion capacity.
Construction performs FFT validation, capability selection, allocation, and
all spread/load/extract/interpolate and FFT pipeline, bind-group, parameter
buffer, and dispatch-grid preparation once. Repeated execution only overwrites
fixed provider buffers, updates retained parameters, and encodes the retained
plans; it performs no transient provider-buffer allocation,
position/deconvolution conversion allocation, pipeline preparation/compilation,
bind-group construction, or provider selection. Output allocation,
non-contiguous 3D mode materialization, and host-device transfers remain
explicit. Type-2 output capacity is `max(mode_count, sample_capacity)`, so
reusable execution cannot underallocate when samples outnumber modes. The
maximum-capacity interpolation grid remains fixed while the shader guards work
with the current logical sample count. The public accelerator boundary accepts
and returns
`hephaestus_wgpu::WgpuDevice`; it exposes no raw device, queue, encoder, or
helper wrapper.

## Performance evidence

The 2026-08-28 reference run used Windows on an Intel Core Ultra 9 285K with
an NVIDIA GeForce RTX 5080 (driver 610.47). The optimized `bench-quick`
`buffer_reuse` suite completed in approximately 162 seconds and retained 100
observations per arm. Values are median milliseconds with the exact 96.4799%
distribution-free median interval in brackets.

| Operation | Per-call construction | Retained buffers | Median ratio |
| --- | ---: | ---: | ---: |
| Type-1 1D, n=256 | 319.778 [318.795, 320.885] | 0.160293 [0.158407, 0.161098] | 1,995.0x |
| Type-2 1D, n=256 | 302.337 [301.849, 303.463] | 0.183750 [0.182199, 0.184924] | 1,645.4x |
| Type-1 3D, 8x8x8 | 478.156 [476.766, 480.485] | 0.125589 [0.124939, 0.126387] | 3,807.3x |
| Type-2 3D, 8x8x8 | 477.974 [476.574, 479.433] | 1.45644 [1.45257, 1.46109] | 328.2x |

The per-call arm constructs all provider buffers and prepared pipelines; the
retained arm performs the same host writes, dispatch, readback, and output
construction. The result isolates plan/pipeline construction from warm NUFFT
execution and is the entry baseline for the grouped correction.

The unchanged 100-observation instrument then measured the grouped candidate.
Values below are warm-path median milliseconds with the same confidence rule;
the delta compares each candidate median with its entry retained-buffer median.

| Operation | Grouped retained buffers | Delta |
| --- | ---: | ---: |
| Type-1 1D, n=256 | 0.151293 [0.149693, 0.153446] | -5.61% |
| Type-2 1D, n=256 | 0.172448 [0.170480, 0.174044] | -6.15% |
| Type-1 3D, 8x8x8 | 0.120422 [0.117019, 0.121540] | -4.11% |
| Type-2 3D, 8x8x8 | 1.37592 [1.37241, 1.38053] | -5.53% |

All four candidate intervals are disjoint from their entry retained-buffer
intervals. The full candidate per-call/retained median ratios are 2,374.6x,
1,860.6x, 4,728.4x, and 370.0x respectively. The absolute per-call medians
varied between uncontrolled host runs, so the correction claim rests on the
warm retained arms. Neither run establishes transfer-free throughput or
performance on a different adapter.

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
The reuse tests execute maximum-capacity and shorter logical requests through
one workspace for all four 1D/3D Type-1/Type-2 paths, comparing every output to
fresh-buffer execution. Host position/deconvolution conversion and
non-contiguous coefficient-refill allocation censuses prove zero
application-owned allocations after retained capacity exists. These tests do
not measure opaque WGPU or driver allocation.

## Compatibility and migration

`apollo-nufft` 0.4.0 removes the public `wgpu_backend` forwarding module and
the public verification-only module. Version 0.5.0 removes the unused
`nufft_wgpu_available` boolean probe: it only erased the acquisition failure
from `NufftWgpuBackend::try_default()`. Consumers import GPU plans, errors,
capabilities, buffers, and `NufftWgpuBackend` directly from the
`apollo_nufft` root, then handle the typed provider acquisition result.
Verification remains private test infrastructure and has no runtime
replacement.

The 2026-08-28 provider cutover also removes `NufftWgpuError::Fft`. The WGPU
path no longer invokes Apollo's CPU/dense-FFT error domain; Hephaestus
preparation, dispatch, allocation, and transfer failures enter through
`NufftWgpuError::Provider`. `NufftGpuBuffers1D::new` and
`NufftGpuBuffers3D::new` replace separate raw geometry arguments with a borrow
of the validated 1D or 3D plan plus `max_samples`. The plan is now the single
source for buffer geometry, deconvolution state, and retained preparation.

## Consequences

The release has no compatibility shim or duplicate transport surface. The
single root export path retains typed Hephaestus ownership, while Leto remains
the host-array boundary. `NufftWgpuBackend::try_default` remains only because
NuFFT must request its transform-specific seven-storage-buffer lower bound
through the provider; it does not implement a device API. `cargo
semver-checks` classifies the removed `Fft` variant as the one expected
`apollo-nufft` major incompatibilities: the removed `Fft` variant and the two
buffer-constructor arity changes. Finite-precision test evidence does not
constitute a machine-checked proof.

The 2026-08-28 revision removes the last dependency on Apollo's former dense
WGPU FFT facade. Hephaestus now owns both the accelerator FFT algorithm and its
prepared resources; Apollo retains NUFFT-specific spread, interpolation,
deconvolution, and host-array semantics.
