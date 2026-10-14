# Quantum Fourier Transform

The QFT is the unitary change of basis at the heart of phase estimation:
it maps each computational basis state to a uniform superposition with
binary-fraction phases. Apollo owns reusable unitary plans for state
vectors in `apollo-qft`, built around `QftPlan`.

```rust
{{#include ../../crates/apollo-qft/examples/book_qft_basis.rs}}
```

## Unitary normalization

Both `forward` and `inverse` scale by `1/sqrt(N)`, so the plans are
unitary: forward maps a basis state to amplitudes of equal magnitude, and
forward-then-inverse is the identity. The sample above checks both facts
on the two-qubit `|00>` state.

## FFT-backed execution

`QftPlan::new` takes a validated `QuantumStateDimension` and builds an
internal FFT plan over the state length. Forward runs the FFT's inverse
kernel and inverse runs its forward kernel (the QFT sign convention),
followed by the unitary scale. State vectors travel as `leto::Array1`
over `eunomia::Complex64`, matching the rest of the Apollo complex path.
