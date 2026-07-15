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

With the `wgpu` feature, Apollo retains the frame layout, Hann window, FFT,
and weighted overlap-add equations. Hephaestus exclusively owns device
acquisition, typed buffers, pipeline and binding preparation, command
recording, submission, synchronization, and transfer. Leto remains the CPU
array boundary.

Radix-2 STFT records the window/pack, bit reversal, butterfly, interleave,
inverse, and overlap-add kernels through typed Hephaestus descriptors.
Bluestein dispatch uses the same provider contract and requests the six
storage bindings required by its two I/O and four chirp-working buffers through
the backend-neutral `DeviceLimits` API. `StftGpuBuffers` retains only typed
provider storage for one fixed radix-2 geometry; parameter blocks and command
streams remain per dispatch.

### Windowed overlap-add theorem

Let `X_m[k]` be the complete length-`N` DFT of the analysis frame
`x[mH + n] w[n]`, with the inverse normalized by `1/N`. The inverse frame
therefore equals `x[mH + n] w[n]` in exact arithmetic by DFT orthogonality.
The synthesis pass multiplies by `w[n]`, and the output sample is

```text
y[t] = sum_m x[t] w[t - mH]^2 / sum_m w[t - mH]^2 = x[t]
```

whenever the denominator is non-zero. Ordered command streams preserve every
producer-before-consumer dependency, including the Bluestein convolution
passes and inverse overlap-add. This is an exact-arithmetic theorem; the
finite-precision GPU result is supported by CPU differential and reconstruction
tests, not a machine-checked proof. ADR 0008 records the ownership and
verification boundary.

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
mismatch rejection. The accelerator suite additionally checks radix-2 and
Bluestein forward/inverse CPU differential behavior, reusable provider storage,
and real-device non-power-of-two execution.
