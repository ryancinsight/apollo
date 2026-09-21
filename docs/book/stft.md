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

The inverse uses weighted overlap-add: each sample is the overlap of the
windowed inverse frames divided by the sum of the squared window over the
frames that cover it. The plan's one window serves both passes, so
reconstruction is exact in exact arithmetic wherever that sum is non-zero (the
nonzero overlap-add condition, weaker than constant overlap-add); the inverse
refuses a plan and length where some sample's sum is at most `ε` of the
largest. That floor is conditioning, not failure: the relative error a small
weight leaves is bounded by `γ √(largest / weight)`, about `7e-7` at `N = 8`,
and only reaches the sample's own magnitude near `weight / largest ≈ γ²`, so
every accepted plan reconstructs to roughly six significant digits. The window is Hann by default, a `Window` family (Hann, Hamming,
Blackman, Tukey) through `StftPlan::with_window`, or caller values through
`StftPlan::with_window_values`.

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
