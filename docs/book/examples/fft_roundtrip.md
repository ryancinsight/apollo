# Example: FFT Round-Trip

**Crate**: `apollo-fft`
**Source**: `crates/apollo-fft/examples/book_fft_roundtrip.rs`

Build a 256-sample dual-tone signal, forward-transform it, find the dominant
frequency bin, and verify the inverse FFT reconstructs the original signal
to machine-epsilon accuracy.

## Source

```rust
{{#include ../../../crates/apollo-fft/examples/book_fft_roundtrip.rs}}
```

## Output

```text
signal length    : 256
spectrum length  : 256
frequency bins   : 256 (0 Hz .. Nyquist)
dominant bin     : 26  (101.6 Hz, magnitude 96.0711)
reconstructed length: 256
max round-trip error: 6.439e-15 (tolerance 1e-11)
FFT round-trip assertions passed
```

## What to notice

- `fft_1d_array` returns a full-length complex spectrum (length = signal
  length).  Positive frequencies occupy bins 0…N/2; negative frequencies
  occupy bins N/2+1…N-1.

- `fftfreq(n, dt)` returns the frequency value for each bin.  Bin `k =
  26` with `dt = 0.001 s` gives `26 / (256 × 0.001) ≈ 101.6 Hz`, the
  nearest grid point to the 100 Hz tone.

- The round-trip error (6.4 × 10⁻¹⁵) is at double-precision machine
  epsilon — the FFT is numerically exact for this input size and kernel.
