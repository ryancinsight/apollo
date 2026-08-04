# Example: Parseval's Theorem

**Crate**: `apollo-fft`
**Source**: `crates/apollo-fft/examples/book_parseval.rs`

Verify Parseval's theorem (`Σ|x[n]|² = (1/N) Σ|X[k]|²`) for a sampled
sinusoid, and confirm the complex FFT round-trip via the in-place API.

## Source

```rust
{{#include ../../../crates/apollo-fft/examples/book_parseval.rs}}
```

## Output

```text
real signal: time energy = 64.000000, spectral energy = 64.000000
complex FFT round-trip max error: 2.486e-15
Parseval + complex FFT assertions passed
```

## What to notice

- A pure sine wave of amplitude 1 sampled over N = 128 points has energy
  N/2 = 64.  Parseval's theorem confirms this exactly: the spectral
  energy Σ|X[k]|²/N = 64.

- The complex FFT in-place API (`fft_1d_complex_inplace` +
  `ifft_1d_complex_inplace`) modifies the array in-place.  The
  forward-then-inverse pair is the identity map; the round-trip error
  (2.5 × 10⁻¹⁵) is at double-precision machine epsilon.

- Apollo uses FFTW-compatible normalisation: the forward FFT does not
  divide by N; the inverse FFT divides by N.  This matches the convention
  used by NumPy, SciPy, and MATLAB.
