# Apollo NUFFT

`apollo-nufft` owns non-uniform Fourier transform logic for Apollo. Dense FFT
execution is delegated to `apollo-fft`; NUFFT-specific spreading,
interpolation, and deconvolution remain in this crate.

## Architecture

```text
src/
  domain/          uniform-domain/grid metadata
  application/     1D and 3D Type-1/Type-2 NUFFT plans
  infrastructure/  Kaiser-Bessel kernels and Fourier-grid helpers
  verification/    analytical and fast/direct parity tests
```

`NufftPlan1D` and `NufftPlan3D` own oversampling shape, kernel width,
Kaiser-Bessel parameters, deconvolution factors, and reusable FFT plans.

Typed execution supports `Complex64`, `Complex32`, and mixed `[f16; 2]`
storage for 1D and 3D Type-1/Type-2 plan surfaces. The Kaiser-Bessel
spreading/interpolation, Apollo FFT execution, and deconvolution path remain
the authoritative `Complex64` implementation; typed APIs convert represented
input into that path and quantize once into caller-owned output storage.

## Accelerator Boundary

The `wgpu` feature selects Hephaestus-backed accelerator execution. Apollo
owns the direct-sum and Kaiser--Bessel algorithms and their WGSL source;
Hephaestus owns device acquisition, typed buffer allocation, binding
validation, pipeline construction, command recording, submission, and
transfer. `NufftWgpuBackend` exposes a `hephaestus_wgpu::WgpuDevice`, never a
raw device or queue.

Direct 1D/3D descriptors bind positions, complex source data, complex output,
and one POD parameter block. Fast descriptors bind the seven canonical
position/value/deconvolution/grid/output buffers plus parameters. The ordered
command stream records spread or load, `GpuFft3d` dispatch, then extract or
interpolate, establishing each device write-before-read dependency. Leto owns
the CPU array/view boundary; accelerator storage is typed `f32` or
`Complex32` and never aliases a Leto allocation.

`NufftGpuBuffers1D` and `NufftGpuBuffers3D` retain provider-owned reusable
storage. Their Type-2 output capacity is `max(mode_count, sample_capacity)`,
so a reusable Type-2 plan remains valid when it evaluates more non-uniform
positions than Fourier modes.

## Mathematical Contract

Type-1 maps non-uniform samples to uniform Fourier modes:

```text
f_k = sum_j c_j exp(-2*pi*i*k*x_j/L)
```

Type-2 maps uniform Fourier modes back to non-uniform positions. Fast execution
uses Kaiser-Bessel spreading/interpolation on an oversampled grid followed by
Apollo FFT execution and deconvolution.

For the direct pair, with the complex inner product, Type-2 is the adjoint of
Type-1:

```text
<Type1(c), f> = <c, Type2(f)>
```

because conjugating the Type-1 negative exponential yields the Type-2 positive
exponential term pointwise. This is an exact-arithmetic theorem. The fast
paths approximate that pair through Kaiser-Bessel gridding. In 1D the inverse
FFT normalizes by the oversampled length `M`; the load stage compensates by
`M` before interpolation to retain the declared unnormalized Type-2
convention. The 3D path retains its declared normalized inverse convention.

## Verification

Tests cover exact DC invariants, fast/direct agreement for fixed inputs,
Kaiser-Bessel non-negativity and peak behavior, `I_0` reference values, Fourier
transform limits, signed index mapping, 3D finite-output behavior, typed
`Complex64` parity, represented-input `Complex32` and mixed `[f16; 2]` parity,
typed Type-2 output parity, shape rejection, and profile/storage mismatch
rejection. Accelerator tests additionally compare direct and fast execution
against CPU references, check invalid plans and reusable-buffer capacities, and
prove bit-exact equality between reusable and non-reusable fast Type-2 output
when the sample count exceeds the mode count. These are empirical
finite-precision checks; no machine-checked proof is claimed.
