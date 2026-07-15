# Apollo Mellin

`apollo-mellin` owns real Mellin moments and log-frequency Mellin spectra over
positive scale-domain signals.

## Architecture

```text
src/
  domain/             positive scale-range contracts and errors
  application/        reusable Mellin plan
  infrastructure/
    kernel/           CPU resampling, moments, and log-frequency kernels
    transport/gpu/
      domain/         GPU capability and error contracts
      application/    concrete-f32 GPU scale plans
      infrastructure/ typed Hephaestus kernels, device boundary, and WGSL
  verification/       analytical integral and contract tests
```

`MellinPlan` is the authoritative scale configuration and execution surface.

Typed execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` input and output storage with owner `f64`
  log-resampling, moment, and spectrum kernels.
- `LOW_PRECISION_F32`: `f32` input and output storage converted through the
  owner path and quantized once into caller-owned real outputs.
- `MIXED_PRECISION_F16_F32`: `f16` input and output storage converted through
  the owner path and quantized once into caller-owned real outputs.

Profile/storage mismatches return `MellinError::PrecisionMismatch`.

With the `wgpu` feature, the accelerator accepts a concrete `f32` scale grid
and domain bounds. Typed input is sealed through `MellinGpuStorage`: native
`f32` input is borrowed at the host boundary and `f16` is explicitly converted
through Mnemosyne scratch; `f64` is rejected at compile time instead of being
silently narrowed. Leto views borrow contiguous input and materialize only
strided logical views. Hephaestus owns all device buffers, typed bindings,
dispatch, synchronization, and transfer; returned Leto arrays are
Mnemosyne-backed.

## Mathematical Contract

For positive scale coordinate `r`, the Mellin moment is

```text
M(s) = int_a^b f(r) r^(s - 1) dr
```

The substitution `r = exp(u)` maps multiplicative scale changes to additive
translations in `u`, enabling Fourier analysis on a logarithmic grid.

## Inverse Transform

`MellinPlan::inverse_spectrum(spectrum, out_min, out_max, output)` inverts the
forward log-frequency spectrum via:

1. **IDFT** of the log-domain spectrum (Moirai-parallel for N >= 256):
   `g[n] = (1/(N·du)) · Re{ sum_k F[k] · exp(+2πi·kn/N) }`
2. **Exp-resample**: linear interpolation of `g` from the log-grid back to the
   linear output domain `[out_min, out_max]`.

`MellinError::SpectrumLengthMismatch` is returned when the spectrum length
differs from the plan sample count.

### Log-grid inverse theorem

Let `g[n]` be the log-resampled signal on a uniform grid of spacing `du`, and
define `F[k] = du sum_n g[n] exp(-2 pi i k n / N)`. The inverse spectrum pass
uses `g[n] = Re(sum_k F[k] exp(+2 pi i k n / N)) / (N du)`. Therefore the two
passes are an inverse pair on the sampled log grid in exact arithmetic.

Proof sketch: substitute the forward sum into the inverse and use DFT
orthogonality, `sum_k exp(2 pi i k (n-m) / N) = N delta_nm`; the factors `du`
and `1/(N du)` cancel. Exponential resampling then reconstructs the linear
grid interpolation of that recovered log-grid signal. The real-device suite
checks the forward result against the CPU implementation and checks a constant
signal through the complete forward/inverse pair.

## Verification

Tests cover constant and power-law analytical integrals, log-resampling
endpoints, uniform resampling, invalid scale contracts, and log-frequency DC
behavior. Typed tests cover `f64`, `f32`, mixed `f16`, represented-input
moments, represented-input spectra, and precision/profile mismatch rejection.
Inverse tests cover constant-signal roundtrip (ε < 1e-10), linear-signal
roundtrip (interpolation error < 0.1 for N=64), wrong-length rejection, and
invalid output-bounds rejection. The all-feature GPU suite covers real-device
CPU forward differential, inverse constant-signal reconstruction, Leto
contiguous/strided views, and typed `f16` input. The `MellinGpuStorage`
compile-fail doctest verifies that `f64` cannot enter the concrete accelerator.
