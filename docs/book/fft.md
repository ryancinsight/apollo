# Fast Fourier Transform

Apollo's FFT implementation provides 1D, 2D, and 3D real and complex transforms
with both runtime-cached and compile-time-fixed plan types.

## Real FFT (rfft)

Transforms real storage (`f64`, `f32`, `F16`) to the complex spectrum of its
plan precision. `fft_1d_array`, `fft_2d_array` and `fft_3d_array` return the
full spectrum, in the input's shape:

```rust,ignore
use apollo_fft::{fft_1d_array, fft_3d_array};

// [N] -> [N] complex
let spectrum: Array1<Complex64> = fft_1d_array(&real_signal);
// [nx, ny, nz] -> [nx, ny, nz] complex
let volume: Array3<Complex64> = fft_3d_array(&real_field);
```

## Half Spectrum

A real input's spectrum repeats itself as conjugates: `X[N-k] = conj(X[k])` in
one dimension, and in three every bin with `k > nz/2` is the conjugate of one
with `k < nz/2`. The half forms keep only the bins that are not repeated, and
compute only those: consecutive pairs of reals are packed as one complex
sample and transformed at half the length, and in three dimensions the x and y
passes then run on the half volume.

```rust,ignore
use apollo_fft::{fft_1d_slice_half_into, ifft_1d_slice_half_into};
use apollo_fft::{fft_3d_array_half_into, ifft_3d_array_half_into};

// [N] -> [N/2 + 1]; the inverse consumes the spectrum as scratch.
fft_1d_slice_half_into(&signal, &mut half);
ifft_1d_slice_half_into(&mut half, &mut signal_back);

// [nx, ny, nz] -> [nx, ny, nz/2 + 1], and back.
fft_3d_array_half_into(&field, &mut half_volume);
ifft_3d_array_half_into(&mut half_volume, &mut field_back);
```

The packing needs a length that is a positive multiple of four; other lengths
are served through the full transform. The 3-D inverse returns the real part of
the full inverse of the half spectrum's Hermitian completion, so imaginary
parts a real field's spectrum cannot have are ignored. `RealFftData` carries the
same pair for callers that hold a plan (`forward_3d_half_into`,
`inverse_3d_half_into`).

## Complex FFT (cfft)

Transforms a complex array to its full complex spectrum:

```rust,ignore
use apollo_fft::{fft_1d_complex, fft_3d_complex};

let out = fft_1d_complex(&complex_signal);
let out = fft_3d_complex(&complex_volume);
```

## Inverse Transforms

`ifft_1d_array` (real), `ifft_1d_complex` (complex), and their 2D/3D
equivalents invert the corresponding forward transforms, normalized by `1/N`.

## Caller-Owned Output

`_into` variants write into a caller-provided output buffer:

```rust,ignore
fft_1d_array_into(&real_signal, &mut output); // no allocation
```

## Leto View Overloads

`_leto` variants accept `ArrayView1` from Leto and return
Mnemosyne-backed Leto arrays:

```rust,ignore
let out = fft_1d_leto(view);       // Leto view -> Leto output
let out = fft_2d_leto(view2d);
```

## Static Plans

`StaticFftPlan1D<F, N>` / `StaticFftPlan2D<F, NX, NY>` / `StaticFftPlan3D<F, NX, NY, NZ>`
embed the precision and the transform size as generics. No plan cache lookup occurs at runtime:

```rust,ignore
use apollo_fft::StaticFftPlan1D;

let plan: StaticFftPlan1D<f64, 4096> = StaticFftPlan1D::new();
plan.forward_complex_inplace(&mut signal);
```
