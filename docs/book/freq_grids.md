# Frequency Grids

Apollo's `fftfreq` and `rfftfreq` utilities construct the
sample-frequency axes that correspond to FFT output arrays.

## 1D Frequency Grid

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_rfftfreq.rs}}
```

## 2D Frequency Grid

A 2-D rfft grid pairs the full `nx` frequencies along x with the
`ny/2 + 1` non-repeated frequencies along y: Apollo halves the last axis.

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_grid2d.rs}}
```

## 3D Frequency Grid

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_grid3d.rs}}
```

## `HalfSpectrum3D`

`HalfSpectrum3D` describes the half spectrum of a 3D real transform. Given a
real shape `[Nx, Ny, Nz]`, the bins a real field's spectrum does not repeat are
`[Nx, Ny, Nz/2 + 1]`: the last axis is the one halved, which is the layout
`fft_3d_array_half_into` writes.

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_halfspec.rs}}
```

## Parseval's Theorem

The energy in the time domain equals the energy in the frequency domain
(up to normalization). Apollo normalizes the inverse by the volume and
leaves the forward unnormalized (`Normalization::FftwCompatible`), so the
time-domain energy equals the spectral energy divided by `N`.
