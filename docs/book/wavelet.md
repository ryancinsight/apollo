# Wavelet Transforms

Apollo provides both discrete (DWT) and continuous (CWT) wavelet transforms
through `apollo-wavelet`.

## Discrete Wavelet Transform (DWT)

```rust
{{#include ../../crates/apollo-wavelet/examples/book_wavelet_dwt.rs}}
```

### Haar Closed Form

The orthonormal Haar stage maps each sample pair to its scaled sum and
difference, which the coefficients expose directly:

```rust
{{#include ../../crates/apollo-wavelet/examples/book_dwt_haar.rs}}
```

### Supported Wavelets

| Wavelet | Description |
|---------|-------------|
| `Haar` | Haar (D2); simplest orthogonal wavelet |
| `Daubechies4` | Daubechies-4 (D4); 4-coefficient orthogonal |

## Continuous Wavelet Transform (CWT)

```rust
{{#include ../../crates/apollo-wavelet/examples/book_wavelet_cwt.rs}}
```

### Supported Analysis Wavelets

| Wavelet | Description |
|---------|-------------|
| `Ricker` | Mexican hat / Ricker wavelet |
| `Morlet { omega0 }` | Real Morlet with angular carrier frequency `omega0` |

## Storage

`DwtCoefficients` and `CwtCoefficients` are Mnemosyne-backed arrays.
`DwtLetoCoefficients` provides Leto view access to DWT outputs.

## Use in ritk

RITK uses Apollo wavelets for multi-resolution medical image analysis
(wavelet-based noise estimation, scale-space decomposition).
