# Parseval's Theorem

Apollo's normalization contracts ensure Parseval's theorem holds across
all transform variants.

## Statement

For a signal `x[n]` of length `N`:

`
Σ_n |x[n]|² = (1/N) · Σ_k |X[k]|²
`

where `X = FFT(x)`. The `1/N` factor can be placed on the forward or
inverse transform.

## `Normalization` Mode

Backends advertise `Normalization::FftwCompatible`: the forward transform
is unnormalized and the inverse carries the `1/N` volume factor, so the
time-domain energy equals the spectral energy divided by `N`.

## Energy Verification

```rust
{{#include ../../crates/apollo-fft/examples/book_fft_parseval_energy.rs}}
```

## Importance for kwavers and ritk

Kwavers uses Parseval's theorem to verify that acoustic energy is conserved
across frequency-domain filtering steps. RITK uses it to validate MR k-space
data consistency.
