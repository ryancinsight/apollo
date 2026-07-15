# ADR 0006: Dense FFT dispatch through Hephaestus

## Status

Accepted for implementation on 2026-07-15.

## Context

The legacy f32 dense-FFT transport owns direct WGPU buffers, bind groups,
pipelines, command encoders, queue submission, and map/readback mechanics.
Those responsibilities duplicate `hephaestus-wgpu` and couple dense-FFT
mathematics to one device API. The obsolete `apollo-wgpu-helpers` dependency
is already removed from this crate, but the raw transport still prevents typed
composition with downstream GPU transforms.

Apollo owns transform definitions, planning, CPU execution, and host-array
boundaries. Hephaestus owns typed accelerator buffers and device execution.
Leto remains a CPU-only host boundary; it does not model GPU buffers.

## Decision

Replace the raw f32 implementation with typed Hephaestus kernel descriptors
and ordered command streams. The f32 plan now accepts `WgpuDevice` and exposes
only `WgpuBuffer<f32>` plus `WgpuCommandStream` at its composition boundary.
Host f64/f16 conversion remains explicit at the Leto boundary before f32
provider storage is written.

The native-f16 implementation is a separate residual scope. It remains inside
its own `f16_plan` hierarchy until it receives the same provider descriptor
implementation; it is not an f32 compatibility path or a fallback.

Apollo retains the dense FFT algorithm and WGSL transform source. Each kernel
descriptor declares bindings and dispatch geometry, while Hephaestus constructs
and owns pipelines, buffers, bindings, encoders, submission, and readback.
The command stream establishes every producer-before-consumer dependency among
axis transforms, transposes, and output transfer.

The provider contract preserves the reusable-buffer API: typed
`ComputeDevice::write_buffer` overwrites an allocated accelerator buffer,
`CommandStream` records barrier-separated axis and chirp passes, and stream
copy operations compose external NUFFT buffers with the plan-owned workspace.
No provider capability change is required.

The f32 migration retains the existing radix-2/radix-4 and Bluestein Chirp-Z
kernel mathematics. All f32 descriptors use one provider-managed binding group:
storage occupies bindings `0..N`, and the POD parameter block occupies the
terminal binding. Pack/unpack consolidates the legacy FFT and volume uniforms
into one `PackParams` block. This is a layout-only change: axis length, batch
count, volume dimensions, axis selector, and workspace length retain their
existing values. One generic pack descriptor owns both directions through an
entry-point marker type.

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

- The f32 plan has no direct WGPU device, queue, pipeline, bind-group, command
  encoder, or transfer ownership. Hephaestus is the sole owner of those
  mechanics and no local compatibility adapter remains.
- Native f16 remains the only direct WGPU residual in `apollo-fft`; the crate
  cannot remove direct `wgpu` and `pollster` dependencies until that scope is
  migrated.
- The f32 host-to-device precision boundary and typed external-buffer contract
  are value-semantic test targets.
- `GpuFft3d::new` now accepts `WgpuDevice` rather than raw device/queue arcs;
  the pre-1.0 release advances to 0.16.0 after semver classification.
