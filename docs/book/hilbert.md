# Hilbert Transform

The Hilbert transform shifts every positive frequency by a quarter turn,
turning a real signal into its analytic companion. Apollo provides Hilbert
through `apollo-hilbert`.

## Definition

For a real signal `x[n]`, the analytic signal is:

`
x_a[n] = x[n] + i·H{x}[n]
`

Its magnitude is the instantaneous envelope, its argument the
instantaneous phase. A pure tone `A·cos(2π·k·n/N)` maps to the phasor
`A·e^{2πi·k·n/N}`: constant envelope `A`, linear phase.

## Properties

- **Quadrature shift**: positive frequencies gain `+i`, negative `−i`
- **Envelope and phase**: `AnalyticSignal` exposes `envelope()`, `phase()`,
  `quadrature()`, and `instantaneous_frequency()` over the same values
- **DC and Nyquist live in the real part**: the quadrature component
  carries no DC (and no Nyquist at even lengths); the analytic signal
  takes its real part from the input instead

## API

```rust
{{#include ../../crates/apollo-hilbert/examples/book_hilbert_envelope.rs}}
```

## Uses

Hilbert is useful in:
- **AM demodulation** — the envelope is the message
- **Nondestructive testing** — ultrasound and seismic envelope detection
- **Phase analysis** — instantaneous frequency of chirps and vibrato
