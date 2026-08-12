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

The `PrecisionProfile` carries a `Normalization` enum:

| Mode | Forward | Inverse | Use |
|------|---------|---------|-----|
| `Backward` (default) | none | 1/N | Standard convention |
| `Forward` | 1/N | none | Analysis-first workflows |
| `Ortho` | 1/√N | 1/√N | Unitary; Parseval without factor |

## Energy Verification

```rust,ignore
use apollo_fft::{fft_1d_array, PrecisionProfile, Normalization};

let profile = PrecisionProfile::with_normalization(Normalization::Ortho);
let spectrum = fft_1d_array_with_profile(&signal, profile)?;

let time_energy:  f64 = signal.iter().map(|x| x * x).sum();
let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
assert!((time_energy - freq_energy).abs() < 1e-10);
```

## Importance for kwavers and ritk

Kwavers uses Parseval's theorem to verify that acoustic energy is conserved
across frequency-domain filtering steps. RITK uses it to validate MR k-space
data consistency.
