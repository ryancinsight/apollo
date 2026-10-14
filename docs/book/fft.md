# Fast Fourier Transform

Apollo's FFT implementation provides 1D, 2D, and 3D real and complex transforms
with both runtime-cached and compile-time-fixed plan types.

## Real FFT (rfft)

Transforms real storage (`f64`, `f32`, `F16`) to the complex spectrum of its
plan precision. `fft_1d_array`, `fft_2d_array` and `fft_3d_array` return the
full spectrum, in the input's shape:

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_rfft.rs}}
```

## Half Spectrum

A real input's spectrum repeats itself as conjugates: `X[N-k] = conj(X[k])` in
one dimension, and in two or three every bin past the half of the last axis
is the conjugate of one below it. The half forms keep only the bins that are
not repeated, and compute only those: consecutive pairs of reals are packed as
one complex sample and transformed at half the length, and in two and three
dimensions the remaining passes then run on the half plane or volume.

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_half.rs}}
```

The packing needs an even length; odd lengths are served through the full
transform. The full-spectrum 1-D, 2-D and 3-D
real entries take the half route where the length admits it, expanding the
half into the full spectrum by copy on the way out and reading only the lower
half on the way back. The 2-D and 3-D inverses return the real part of the full
inverse of the half spectrum's Hermitian completion, so imaginary parts a real
field's spectrum cannot have are ignored. `RealFftData` carries the same pairs
for callers that hold a plan (`forward_2d_half_into`, `inverse_2d_half_into`,
`forward_3d_half_into`, `inverse_3d_half_into`).

## Complex FFT (cfft)

Transforms a complex array to its full complex spectrum:

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_cfft.rs}}
```

## Inverse Transforms

`ifft_1d_array` (real), `ifft_1d_complex` (complex), and their 2D/3D
equivalents invert the corresponding forward transforms, normalized by `1/N`.

## Caller-Owned Output

`_into` variants write into a caller-provided output buffer:

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_into.rs}}
```

## Leto View Overloads

`_leto` variants accept `ArrayView1` from Leto and return
Mnemosyne-backed Leto arrays:

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_leto.rs}}
```

## Static Plans

`StaticFftPlan1D<F, N>` / `StaticFftPlan2D<F, NX, NY>` / `StaticFftPlan3D<F, NX, NY, NZ>`
embed the precision and the transform size as generics. No plan cache lookup occurs at runtime:

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_static.rs}}
```
