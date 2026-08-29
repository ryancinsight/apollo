# Apollo STFT

`apollo-stft` owns short-time Fourier transform planning and execution for
Apollo.

## Architecture

```text
src/
  domain/          frame, hop, and execution error contracts
  application/     reusable STFT plan and overlap-add execution
  infrastructure/  CPU convenience wrappers
```

`StftPlan` is the single source of truth for frame length, hop length, Hann
window coefficients, frame count, and the backing Apollo FFT plan.

## Mathematical Contract

Forward STFT uses centered frames. Each frame is multiplied by the Hann window
and transformed by the Apollo FFT plan. Inverse STFT applies the inverse frame
FFT, multiplies by the same window, overlap-adds, and divides each sample by
the accumulated squared-window weight.

For every sample with non-zero weight,

```text
sum_m x[t] w[t - mH]^2 / sum_m w[t - mH]^2 = x[t]
```

This gives exact reconstruction in exact arithmetic for covered samples.

## Accelerator Execution

With the `wgpu` feature, Apollo owns the frame layout, Hann window, spectrum
conversion, synthesis window, and weighted overlap-add equations. Hephaestus
owns every dense FFT and Bluestein algorithm, device, typed buffer, pipeline,
binding, command, submission, synchronization, and transfer contract. Leto
remains the CPU array boundary.

`StftGpuBuffers` retains a dense `[frame_count, frame_len]` split-complex frame
plane and two prepared Hephaestus plans with active axis `[1]`. Each row is an
independent frame FFT; the frame axis is never transformed. The same workspace
retains the Apollo-domain pack, interleave, deinterleave, synthesis-window, and
overlap-add dispatches plus host upload/readback capacity. Power-of-two and
non-power-of-two frame lengths use this one path; Apollo contains no private
bit-reversal, butterfly, or Bluestein shader.

Forward execution encodes pack/window → FFT → interleave in one command stream.
Inverse execution encodes deinterleave → inverse FFT → synthesis window →
overlap-add in one command stream. Hephaestus applies inverse normalization by
`1/frame_len`, so Apollo does not scale the inverse a second time. The STFT
domain kernels fit the default provider limits; the obsolete six-storage-buffer
request is removed.

### Windowed overlap-add theorem

Let `X_m[k]` be the complete length-`N` DFT of the analysis frame
`x[mH + n] w[n]`, with the inverse normalized by `1/N`. The inverse frame
therefore equals `x[mH + n] w[n]` in exact arithmetic by DFT orthogonality.
The synthesis pass multiplies by `w[n]`, and the output sample is

```text
y[t] = sum_m x[t] w[t - mH]^2 / sum_m w[t - mH]^2 = x[t]
```

whenever the denominator is non-zero. Ordered command streams preserve every
producer-before-consumer dependency, including the provider-owned
non-power-of-two transform and inverse overlap-add. This is an exact-arithmetic
theorem; the finite-precision GPU result is supported by CPU differential and
reconstruction tests, not a machine-checked proof. ADR 0008 records the
ownership and verification boundary.

## Execution Surfaces

- `forward` and `inverse` allocate returned arrays.
- `forward_into` and `inverse_into` use caller-owned output buffers. Inverse
  overlap-add execution reuses per-thread frame, complex, overlap, and weight
  workspaces.
- `forward_typed_into` and `inverse_typed_into` support Apollo precision
  profiles without duplicating frame or FFT kernels. Typed execution reuses
  per-thread f64/Complex64 bridge workspaces instead of allocating bridge
  arrays per call.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` signal storage and `Complex64` spectrum storage.
- `LOW_PRECISION_F32`: `f32` signal storage and `Complex32` spectrum storage,
  converted through the owner path and quantized once into caller-owned output.
- `MIXED_PRECISION_F16_F32`: `f16` signal storage and `[f16; 2]` spectrum
  storage, converted through the owner path and quantized once into
  caller-owned output.

Profile/storage mismatches return `StftError::PrecisionMismatch`.

## Verification

The crate verifies Hann symmetry, forward/inverse reconstruction,
caller-owned forward and inverse parity, invalid configuration rejection,
short-input rejection, and property-based reconstruction over deterministic
signals, inverse workspace reuse, and caller-owned forward parity. Typed tests
cover `f64`, `f32`, mixed `f16`, represented-input spectrum parity, `f32`
inverse roundtrip, repeated typed workspace reuse, and precision/profile
mismatch rejection. The accelerator suite additionally checks every bin of
distinct-row selected-axis direct-DFT cases at lengths 8 and 6, inverse
normalization, reusable provider and host storage across two different inputs,
pre-mutation geometry/device rejection, and real-device non-power-of-two
execution.
