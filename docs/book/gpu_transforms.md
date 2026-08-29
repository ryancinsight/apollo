# GPU Transforms

Apollo owns transform-domain mathematics. Hephaestus owns accelerator devices,
typed buffers, prepared pipelines, command streams, and dense WGPU FFT
algorithms. Leto supplies the shape and stride metadata shared by CPU and GPU
operands.

## Dense FFT

A dense accelerator FFT is prepared through Hephaestus rather than an Apollo
backend wrapper. The rank is encoded by `Layout<R>` and `FftOperands`. Upload
the real component and allocate the imaginary component through `WgpuDevice`,
construct a C-contiguous Leto layout, then pass the two `StridedView` operands
to `WgpuFftOps::prepare_fft`. Retain the returned plan and execute it with
`WgpuFftOps::dispatch_fft`, or encode it into a larger command stream with the
`FftOps` composition surface.

Preparation validates shape, strides, aliasing, scalar capability, and device
ownership before mutation. It also compiles and binds the required radix or
Bluestein pipelines. Retain the prepared plan with its fixed buffers; repeated
dispatch then performs no plan allocation or compilation.

Ranks one through three use the same API. The buffer scalar selects f32 or
native-f16 arithmetic; native f16 requires a `ShaderF16`-qualified device.

## Composition

`FftOps::encode_fft` records a prepared transform into an existing Hephaestus
command stream. Apollo NUFFT uses this boundary to order spread/load, FFT, and
extract/interpolate passes without an intermediate submission or host copy.
Forward and inverse plans live beside the reusable oversampled grids.

Apollo STFT prepares a rank-two frame plane with shape
`[frame_count, frame_len]` and active axis `[1]`. `prepare_fft_axes` retains
the frame axis as a batch dimension, so every row is transformed independently.
Apollo encodes its Hann pack/interleave or deinterleave/window/overlap-add
kernels around the retained Hephaestus plan in one grouped command stream.
Power-of-two and non-power-of-two frame lengths share this path; Bluestein
state remains inside Hephaestus.

## Domain-specific transforms

Apollo's `apollo-fft::WgpuTransformBackend<K>` remains the shared scaffold for
non-FFT transform kernels. A transform crate supplies its own zero-sized
planner and executor implementation, while this Apollo scaffold centralizes
typed storage, capability reporting, validation, and provider errors. It is
not a Hephaestus API or a dense FFT backend and does not duplicate Hephaestus
`FftOps`.

Accelerator implementations are verified against the corresponding Apollo CPU
operator under a derived finite-precision bound. Provider absence is reported
at acquisition; a present provider failure is never converted into a silent CPU
fallback.
