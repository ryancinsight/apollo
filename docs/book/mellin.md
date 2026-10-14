# Mellin Transform

The Mellin transform analyzes positive scale-domain signals in
log-frequency: real moments summarize the distribution, and the spectrum
exposes scale-invariant structure the Fourier transform spreads out.
Apollo owns the moments and spectra in `apollo-mellin`, built around
`MellinPlan`.

```rust
{{#include ../../crates/apollo-mellin/examples/book_mellin_moment.rs}}
```

## Moments

`MellinPlan::moment` evaluates `M(s) = int f(r) r^(s-1) dr` over the
signal's domain with the trapezoid rule, which is exact on linear
integrands: the sample checks the unit signal against its closed form
`M(2) = 3/2`. Long signals take a SIMD dot path past the implementation
threshold.

## Log-grid spectra

`forward_resample` moves a linear-domain signal onto the plan's
logarithmic scale grid, and `forward_spectrum` follows with the
log-frequency spectrum over that grid. The `f64` resampling kernel stays
the owner implementation across storage profiles; both entry points
validate the signal domain up front, and the Leto-view variants borrow
contiguous inputs directly,
copying once only for strided views.
