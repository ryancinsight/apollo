# Frequency Grids

Apollo's `freq_grids` utilities construct the sample-frequency axes that
correspond to FFT output arrays.

## 1D Frequency Grid

```rust,ignore
use apollo_fft::freq_grid_1d;

// Returns [N/2+1] sample frequencies in cycles/sample (rfft convention)
let freqs = freq_grid_1d(N, sample_rate)?;  // in Hz if sample_rate is Hz
```

## 2D Frequency Grid

```rust,ignore
let (fx, fy) = freq_grid_2d(Nx, Ny, dx, dy)?;
// fx: [Nx/2+1, Ny]  fy: [Nx/2+1, Ny]  (rfft convention on x-axis)
```

## 3D Frequency Grid

```rust,ignore
let (fx, fy, fz) = freq_grid_3d(Nx, Ny, Nz, dx, dy, dz)?;
```

## `HalfSpectrum3D`

`HalfSpectrum3D` carries the output shape of a 3D real FFT,
exploiting Hermitian symmetry. Given input shape `[Nx, Ny, Nz]`,
the half-spectrum shape is `[Nx/2+1, Ny, Nz]`:

```rust,ignore
let hs = HalfSpectrum3D::from_input_shape([256, 256, 256])?;
// hs.shape() == [129, 256, 256]
```

## Parseval's Theorem

The energy in the time domain equals the energy in the frequency domain
(up to normalization). Apollo's normalization modes (`Normalization::Forward`,
`Normalization::Backward`, `Normalization::Ortho`) control how the 1/N
factor is distributed between forward and inverse transforms to satisfy the
Parseval contract.
