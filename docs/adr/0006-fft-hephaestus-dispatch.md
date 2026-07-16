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

Replace raw dense-FFT transport with typed Hephaestus kernel descriptors and
ordered command streams. `GpuFft3d<T>` owns the storage-generic plan; its
sealed `FftStorage` contract selects the WGSL source, physical coefficient
encoding, and available radix entries. The f32 composition boundary exposes
only `WgpuBuffer<f32>` plus `WgpuCommandStream`. Native half storage uses
`WgpuBuffer<u16>` for IEEE-754 bit patterns while its WGSL declares
`array<f16>`. Host conversion remains explicit at the Leto boundary.

Both storage representations use the same typed allocation, pipeline,
binding, command-stream, submission, and readback implementation. Native half
constructors accept or acquire `WgpuDevice` with the provider-owned required
`ShaderF16` contract. The half shader provides radix-two entries only, which
the storage capability selects at plan construction; this is not an f32
fallback or a second dispatcher.

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

For the native-half all-Bluestein 3×3×3 reconstruction fixture, each forward
axis has 43 half-rounding sites and each inverse axis has 45, including
twiddle narrowing, radix arithmetic, and normalization. Together with input
quantization this gives \(k=1+3(43+45)=265\). The test asserts
\(\gamma_{265}\lVert x\rVert_1\), where
\(\gamma_k=ku/(1-ku)\) and \(u=2^{-11}\). This is an analytical fixture
bound; real-device roundtrip remains empirical evidence.

## Consequences

- The f32 plan has no direct WGPU device, queue, pipeline, bind-group, command
  encoder, or transfer ownership. Hephaestus is the sole owner of those
  mechanics and no local compatibility adapter remains.
- Native half storage has no direct WGPU, Pollster, pipeline, binding,
  command-stream, or readback ownership in `apollo-fft`; it reuses the sealed
  typed storage implementation.
- The f32 host-to-device precision boundary and typed external-buffer contract
  are value-semantic test targets.
- `GpuFft3d::new` now accepts `WgpuDevice` rather than raw device/queue arcs;
  the pre-1.0 release advances to 0.16.0 after semver classification.
- `GpuFft3dF16Native::try_from_device` is the native-half construction
  boundary. Its caller acquires a typed Hephaestus device with
  `DeviceFeature::ShaderF16`; the feature is mandatory, not an optional
  fallback.
