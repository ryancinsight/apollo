# Apollo CZT

`apollo-czt` owns the chirp z-transform implementation for Apollo.

## Architecture

```text
src/
  domain/          errors and validated contracts
  application/     reusable CZT plan and execution policy
  infrastructure/  CPU convenience transport
```

The application plan is the single source of truth for transform lengths,
spiral parameters, chirp factors, the convolution kernel, and the backing
Apollo FFT plan.

Leto owns host arrays and strided views. Mnemosyne owns reusable CPU/GPU bridge
scratch and Leto result storage. Hermes supplies runtime-selected SIMD complex
row reductions, while Moirai schedules direct rows and Bluestein buffer stages.
Under the `wgpu` feature, Apollo retains the CZT WGSL formulas and Hephaestus
owns typed device buffers, kernel preparation, command streams, submission,
synchronization, and transfers; `apollo-czt` has no direct WGPU dependency.

## Precision Contract

`forward_typed_into` supports:

- `HIGH_ACCURACY_F64`: `Complex64` storage.
- `LOW_PRECISION_F32`: `Complex32` storage.
- `MIXED_PRECISION_F16_F32`: `[f16; 2]` storage, where lane 0 is the real
  component and lane 1 is the imaginary component.

All typed paths reuse the authoritative `Complex64` Bluestein path and quantize
once at the storage boundary. Profile/storage mismatch is rejected with
`CztError::PrecisionMismatch`.

## Mathematical Contract

For input `x[0..n)`, output length `m`, and non-zero finite spiral parameters
`a` and `w`, the transform is

```text
X[k] = sum_n x[n] a^-n w^(n k)
```

`forward_direct` evaluates this definition directly in `O(nm)` time.
`forward` and `forward_into` use Bluestein's identity
`nk = (n^2 + k^2 - (k-n)^2) / 2` to reduce the transform to one zero-padded
convolution evaluated by `apollo-fft`.

## Inverse Transform

`CztPlan::inverse(spectrum)` inverts the square (M == N) CZT by solving the
Vandermonde system

```text
V y = X   where V[k,n] = W^(kn)
```

via the Björck-Pereyra algorithm (O(N²), in-place Newton divided-difference
phasors). After solving for `y`, the original signal is recovered as
`x[n] = y[n] · A^n`.

`CztError::NotInvertible` is returned when:
- the plan is non-square (M ≠ N), or
- a Vandermonde node denominator falls below `f64::EPSILON × 1024` (node
  collision — W is a root of unity of order ≤ N).

## Verification

The crate validates:

- direct and Bluestein paths agree on analytical complex inputs
- invalid lengths and mismatched execution buffers are rejected
- caller-owned output matches the allocating fast path
- typed `Complex64`, `Complex32`, and mixed `[f16; 2]` storage paths match the
  allocating fast path within analytically derived storage bounds
- inverse roundtrip at DFT parameters, general `A` offset, non-unit `W` spacing
- inverse rejects non-square plans and wrong-length spectrum inputs
- Hephaestus forward execution matches the CPU direct formula and the exact
  unit-impulse oracle when a WGPU device is available
- Hephaestus DFT-parameter forward/inverse execution satisfies the derived
  `f32` roundtrip bound
- Leto contiguous, strided, typed, and inverse boundaries preserve values while
  computing or downloading directly into Mnemosyne-backed output storage

Production CZT code depends on Apollo-owned FFT kernels. External FFT engines
belong only in `apollo-validation`.
