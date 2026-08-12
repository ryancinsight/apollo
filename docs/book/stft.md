# STFT — Short-Time Fourier Transform

The short-time Fourier transform (STFT) segments a signal into overlapping
frames and applies a windowed FFT to each, producing a time-frequency representation.

## `StftPlan`

```rust,ignore
use apollo_stft::{StftPlan, stft, istft};

let plan = StftPlan::new(
    window_length: 512,
    hop_size:      256,
    fft_size:      512,   // must be >= window_length
    window:        Window::Hann,
)?;
```

## Forward Transform

```rust,ignore
// signal: [T]  ->  spectrum: [frames, fft_size/2+1] complex
let spectrum = stft(&signal, &plan)?;
```

## Inverse Transform

```rust,ignore
// spectrum: [frames, fft_size/2+1]  ->  signal: [T]
let reconstructed = istft(&spectrum, &plan)?;
```

The inverse uses the overlap-add method for reconstruction. Perfect
reconstruction holds when the window satisfies the COLA constraint
(constant overlap-add).

## Leto Overloads

`stft_leto` and `istft_leto` accept Leto `ArrayView1` and return
Leto arrays for zero-copy integration with the array storage layer.

## Use in kwavers

Kwavers uses Apollo STFT for ultrasound RF signal analysis and spectral
feature extraction in acoustic simulation workflows.
