# Apollo SDFT

`apollo-sdft` owns sliding DFT state for fixed-window streaming signals.

## Architecture

```text
src/
  domain/          window-length and bin-count contracts
  application/     reusable plan and streaming state
  infrastructure/  direct initialization and recurrence kernels
  verification/    direct-vs-recurrence and contract tests
```

`SdftPlan` owns validated window metadata and twiddle factors. `SdftState` owns
the current window, tracked bins, and update count.

Typed direct-bin execution uses Apollo's shared precision profile contract:

- `HIGH_ACCURACY_F64`: `f64` window storage and `Complex64` bin storage.
- `LOW_PRECISION_F32`: `f32` window storage and `Complex32` bin storage,
  converted through the owner path and quantized once into caller-owned output.
- `MIXED_PRECISION_F16_F32`: `f16` window storage and `[f16; 2]` bin storage,
  converted through the owner path and quantized once into caller-owned output.

Profile/storage mismatches return `SdftError::PrecisionMismatch`.

Typed direct-bin execution reuses per-thread f64/Complex64 bridge workspaces
and routes arithmetic through the same direct-bin owner kernel as f64 callers.

## Hephaestus Accelerator Boundary

The optional `wgpu` feature uses Hephaestus typed buffers, kernel interfaces,
and command streams. Apollo contains the SDFT kernel descriptors and WGSL
formulae; it does not construct device buffers, pipelines, bind groups,
encoders, submissions, or readbacks directly.

The accelerator has a deliberately narrower representation contract than the
CPU plan:

- native real `f32` input and `Complex32` bin output transfer directly;
- reduced `f16` / `[f16; 2]` storage converts once through Mnemosyne scratch;
- CPU `f64` / `Complex64` storage is not admitted by `SdftGpuRealStorage` and
  `SdftGpuBinStorage`, preventing an implicit precision-changing dispatch.

Leto views are the host boundary: contiguous views borrow with `Cow`, strided
views materialize their logical order once, and generated results use
Mnemosyne-backed Leto storage. The old `wgpu_backend` forwarding module and
raw device/queue accessors are removed; `SdftWgpuBackend` exposes the
Hephaestus device abstraction only.

## Mathematical Contract

For a window of length `N`, each update removes `x_old`, appends `x_new`, and
updates tracked bins by the sliding recurrence derived from the DFT definition.
The state is equivalent to recomputing direct DFT bins over the current window
after every update.

### Complete-bin inversion theorem

For the full DFT, let

`X[k] = Σ_{n=0}^{N-1} x[n] exp(-2πikn/N)`.

The inverse kernel computes

`x̂[m] = (1/N) Σ_{k=0}^{N-1} X[k] exp(+2πikm/N)`.

Substituting the forward definition yields

`x̂[m] = Σ_n x[n] · (1/N) Σ_k exp(2πik(m-n)/N)`.

The finite root-of-unity sum equals one when `m = n` and zero otherwise, so
`x̂[m] = x[m]` in exact arithmetic. Consequently, accelerator inverse dispatch
requires `bin_count == window_len`. A partial set of tracked bins is useful for
forward SDFT analysis but is a projection, not an invertible spectrum.

## Verification

Tests cover initial direct-bin equivalence, update recurrence equivalence,
zero-state behavior, update counting, invalid contracts, and direct DFT parity
after a full window of pushes. Typed tests cover `f64`, `f32`, mixed `f16`,
represented-input direct-bin parity, repeated workspace reuse, caller-owned
output reuse, output length rejection, and precision/profile mismatch
rejection. GPU tests cover real-device CPU differential forward values,
complete-bin round trips, Leto contiguous and strided views, reduced storage,
and partial-inverse rejection.
