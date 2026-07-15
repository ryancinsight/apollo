# ADR 0006: Dense FFT dispatch through Hephaestus

## Status

Accepted for implementation on 2026-07-15.

## Context

`apollo-fft` owns direct WGPU device acquisition, buffers, bind groups,
pipelines, command encoders, queue submission, and map/readback mechanics.
Those responsibilities duplicate `hephaestus-wgpu` and retain the obsolete
`apollo-wgpu-helpers` wrapper. They also couple dense-FFT mathematics to one
device API.

Apollo owns transform definitions, planning, CPU execution, and host-array
boundaries. Hephaestus owns typed accelerator buffers and device execution.
Leto remains a CPU-only host boundary; it does not model GPU buffers.

## Decision

Replace the raw f32 and native reduced-precision WGPU implementations with
typed Hephaestus kernel descriptors and ordered command streams. A sealed
accelerator storage trait admits only the concrete storage profiles implemented
by the provider. Host conversion validates representability before a provider
allocation or dispatch; any lossy reduced-precision conversion is an explicit
operation.

Apollo retains the dense FFT algorithm and WGSL transform source. Each kernel
descriptor declares bindings and dispatch geometry, while Hephaestus constructs
and owns pipelines, buffers, bindings, encoders, submission, and readback.
The command stream establishes every producer-before-consumer dependency among
axis transforms, transposes, and output transfer.

The provider contract already preserves the reusable-buffer API: its typed
`ComputeDevice::write_buffer` overwrites an allocated accelerator buffer without
exposing WGPU. `CommandStream` records barrier-separated axis and chirp passes,
and `GroupedKernelDevice` represents the pack/unpack shader's existing three
binding groups. Therefore this migration requires no provider capability change.

The f32 migration retains the existing radix-2/radix-4 and Bluestein Chirp-Z
kernel mathematics. FFT and Chirp-Z descriptors bind their storage in group
zero and their typed parameter block in group one. Pack/unpack currently carry
two raw uniform blocks; their provider-native form consolidates those values
into one `PackParams` POD block at the volume group. This is a layout-only
change: the axis length, batch count, volume dimensions, axis selector, and
workspace length retain their existing values. One grouped descriptor then
becomes the SSOT for both packing directions.

## Mathematical contract

For dimensions \(N_x,N_y,N_z\), the forward three-dimensional discrete Fourier
transform is

\[
X_{k_x,k_y,k_z} = \sum_{x=0}^{N_x-1}\sum_{y=0}^{N_y-1}\sum_{z=0}^{N_z-1}
x_{x,y,z}e^{-2\pi i(k_xx/N_x+k_yy/N_y+k_zz/N_z)}.
\]

The inverse uses the positive exponent and normalization
\((N_xN_yN_z)^{-1}\). Orthogonality of roots of unity gives
\(\mathcal{F}^{-1}(\mathcal{F}(x))=x\) in exact arithmetic. This is a proof
sketch of the mathematical contract, not a machine-checked proof. CPU/GPU
differential and inverse-roundtrip tests provide empirical finite-precision
evidence after migration.

## Consequences

- Apollo removes its direct WGPU and helper dependency edges from this crate.
- Hephaestus is the sole owner of device mechanics; no local compatibility
  adapter remains.
- The host-to-device precision boundary is explicit and testable.
- The public GPU surface may change; the pre-1.0 release advances to 0.16.0
  after semver classification.
