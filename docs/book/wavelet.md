# Wavelet Transforms

Apollo provides both discrete (DWT) and continuous (CWT) wavelet transforms
through `apollo-wavelet`.

## Discrete Wavelet Transform (DWT)

```rust,ignore
use apollo_wavelet::{DwtPlan, DiscreteWavelet};

let plan = DwtPlan::new(
    wavelet: DiscreteWavelet::Daubechies4,
    levels:  5,
    signal_length: 1024,
)?;

let coeffs = plan.forward(&signal)?;  // DwtCoefficients
let reconstructed = plan.inverse(&coeffs)?;
```

### Supported Wavelets

| Wavelet | Description |
|---------|-------------|
| `Haar` | Haar (D2); simplest orthogonal wavelet |
| `Daubechies4` | Daubechies-4 (D4); 4-coefficient orthogonal |

## Continuous Wavelet Transform (CWT)

```rust,ignore
use apollo_wavelet::{CwtPlan, ContinuousWavelet};

let plan = CwtPlan::new(
    wavelet:     ContinuousWavelet::Morlet,
    scale_min:   1.0,
    scale_max:   128.0,
    num_scales:  64,
    signal_length: 1024,
)?;

let scalogram = plan.forward(&signal)?;  // CwtCoefficients: [scales, time]
```

### Supported Analysis Wavelets

| Wavelet | Description |
|---------|-------------|
| `Ricker` | Mexican hat / Ricker wavelet |
| `Morlet` | Complex Morlet; frequency-localized |

## Storage

`DwtCoefficients` and `CwtCoefficients` are Mnemosyne-backed arrays.
`DwtLetoCoefficients` provides Leto view access to DWT outputs.

## Use in ritk

RITK uses Apollo wavelets for multi-resolution medical image analysis
(wavelet-based noise estimation, scale-space decomposition).
