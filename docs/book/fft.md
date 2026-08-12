# Fast Fourier Transform

Apollo's FFT implementation provides 1D, 2D, and 3D real and complex transforms
with both runtime-cached and compile-time-fixed plan types.

## Real FFT (rfft)

Transforms a real array to its half-spectrum complex representation,
exploiting Hermitian symmetry:

```rust,ignore
use apollo_fft::{fft_1d_array, fft_2d_array};

// 1D real FFT: [N] -> [N/2+1] complex
let spectrum: Array1<Complex64> = fft_1d_array(real_signal)?;

// 2D real FFT
let spec2d: Array2<Complex64> = fft_2d_array(real_field)?;
```

## Complex FFT (cfft)

Transforms a complex array to its full complex spectrum:

```rust,ignore
use apollo_fft::{cfft_1d_array, cfft_3d_array};

let out = cfft_1d_array(complex_signal)?;
let out = cfft_3d_array(complex_volume)?;
```

## Inverse Transforms

`ifft_1d_array` (real), `icfft_1d_array` (complex), and their 2D/3D
equivalents invert the corresponding forward transforms.

## In-Place Variants

`_into` variants write into a caller-provided output buffer:

```rust,ignore
fft_1d_array_into(&real_signal, &mut output)?;  // avoids allocation
```

## Leto View Overloads

`_leto` variants accept `ArrayView1` from Leto and return
Mnemosyne-backed Leto arrays:

```rust,ignore
fft_1d_leto(view)?       // Leto view -> Leto output
fft_2d_leto(view2d)?
```

## Static Plans

`StaticFftPlan1D<N>` / `StaticFftPlan2D<NX, NY>` / `StaticFftPlan3D<NX, NY, NZ>`
embed the transform size as a const-generic. No plan cache lookup occurs at runtime:

```rust,ignore
use apollo_fft::StaticFftPlan1D;

let plan: StaticFftPlan1D<4096> = StaticFftPlan1D::new();
let out = plan.forward(&input)?;
```
