# STFT — Short-Time Fourier Transform

The short-time Fourier transform (STFT) segments a signal into overlapping
frames and applies a windowed FFT to each, producing a time-frequency representation.

## `StftPlan`

```rust,ignore
use apollo_stft::StftPlan;

let plan = StftPlan::new(512, 256)?;
```

## Forward Transform

```rust,ignore
// signal: [f64] -> frame_count * frame_len complex bins
let spectrum = plan.forward(&signal)?;
```

## Inverse Transform

```rust,ignore
let reconstructed = plan.inverse(&spectrum, signal.len())?;
```

The inverse uses the overlap-add method for reconstruction. Perfect
reconstruction holds when the window satisfies the COLA constraint
(constant overlap-add).

## GPU execution

With the `wgpu` feature, `StftWgpuBackend` composes Apollo framing and Hann/WOLA
kernels around two retained Hephaestus plans. The dense frame plane has shape
`[frame_count, frame_len]`; only axis 1 is transformed, so rows remain
independent. Both radix and non-power-of-two lengths use the same provider
surface. Reuse one `StftBuffers` value for repeated calls of the same geometry
to retain GPU preparation and host transfer capacity.

## Leto Overloads

`stft_leto` and `istft_leto` accept Leto `ArrayView1` and return
Leto arrays for zero-copy integration with the array storage layer.

## Use in kwavers

Kwavers uses Apollo STFT for ultrasound RF signal analysis and spectral
feature extraction in acoustic simulation workflows.
