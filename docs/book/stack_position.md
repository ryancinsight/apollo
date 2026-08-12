# Position in the Stack

## What Apollo Owns

Apollo is the Atlas spectral transform layer. It owns:

- **FFT** — 1D/2D/3D real/complex transforms, plan cache, GPU backend
- **STFT** — short-time Fourier transform and inverse
- **Wavelet transforms** — DWT (Haar, Daubechies) and CWT (Ricker, Morlet)
- **NTT** — number theoretic transform over finite fields
- **DHT** — discrete Hartley transform
- **Frequency grids** — sample-frequency axis construction
- **Parseval normalization contracts** — energy conservation across transforms
- **GPU transform backend** — wgpu-based FFT dispatch

Apollo does **not** own acoustic physics (kwavers), MR imaging (ritk),
dose computation (helios), tensor algebra (coeus), or memory allocation.

## Where Apollo Sits

`	ext
eunomia  +  leto (array storage)
  |
  v
apollo (spectral transforms)
  |                        |
  v                        v
kwavers (ultrasound)     ritk (MRI k-space)
coeus (spectral NNs)     helios (CT reconstruction)
`

## Consumers

| Consumer | How Apollo is used |
|----------|--------------------|
| `kwavers` | Acoustic pressure field FFT, STFT spectral analysis |
| `ritk` | MR k-space FFT, wavelet multi-resolution analysis |
| `helios` | CT reconstruction (filtered back-projection, frequency-domain filtering) |
| `coeus` | Spectral neural network layers (`coeus-fft`) |
| `CFDrs` | Spectral Poisson solver for incompressible flow |

## Hephaestus Integration

GPU transforms use `WgpuTransformBackend` from `hephaestus-wgpu`.
The shared validation helpers in `WgpuTransformBackend` are consumed
by `apollo-fft`, `apollo-gft`, and `apollo-validation`.

## Leto Integration

`_leto` variant functions accept Leto `ArrayView` inputs and return
Mnemosyne-backed Leto output arrays. This avoids unnecessary copies at
the Apollo/Leto boundary.
