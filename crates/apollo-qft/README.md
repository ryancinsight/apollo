# Apollo QFT

`apollo-qft` owns reusable unitary quantum Fourier transform plans for Apollo
state vectors.

## Architecture

```text
src/
  domain/          quantum state-dimension contracts and errors
  application/     reusable QFT plan
  infrastructure/  dense reference oracle and accelerator execution
  verification/    unitarity, roundtrip, and property tests
```

`QftPlan` retains the state dimension and one shared Apollo FFT plan. Plan
clones share that immutable provider state, and warmed caller-owned forward and
inverse execution allocates no storage. The public dense Hermes/Moirai kernel
remains available as the independent direct-transform oracle; it is not the
reusable plan's execution route. Serialization emits the historical
`dimension`/`twiddles` representation without retaining a twiddle table in the
plan.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `Complex64` storage with the owner unitary FFT route.
- `LOW_PRECISION_F32`: `Complex32` storage converted through the owner path and
  quantized once into caller-owned output.
- `MIXED_PRECISION_F16_F32`: `[f16; 2]` real/imaginary lane storage converted
  through the owner path and quantized once into caller-owned output.

Profile/storage mismatches return `QftError::PrecisionMismatch`.

## Mathematical Contract

For state vector `x` of length `N`, the unitary QFT is

```text
X[k] = N^(-1/2) sum_j x[j] exp(2*pi*i*j*k/N)
```

The inverse uses the conjugate phase. The transform preserves norm because its
matrix columns are orthonormal.

## Hephaestus Accelerator Contract

Apollo owns the unitary QFT formula and WGSL source; Hephaestus owns typed
`Complex32` buffers, parameter upload, binding validation, dispatch, and
readback. The concrete accelerator admits native `Complex32` plus explicit
`[f16; 2]` host promotion, and rejects `Complex64` at compile time rather than
silently narrowing it. The unitary identity is documented mathematics; the
real-device CPU differential and inverse-roundtrip suite provides executable
evidence rather than a machine-checked proof.

## Verification

Tests cover two-point analytical output, norm preservation, inverse roundtrip,
in-place/convenience parity, invalid contracts, unitary matrix columns, `N=1`,
non-power-of-two `N=3`, and randomized vectors. Typed tests cover `Complex64`,
`Complex32`, mixed `[f16; 2]`, caller-owned forward/inverse parity, inverse
roundtrip, and precision/profile mismatch rejection.
