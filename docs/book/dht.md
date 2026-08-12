# Discrete Hartley Transform

The Discrete Hartley Transform (DHT) is a real-valued analogue of the DFT.
Apollo provides DHT through `apollo-dht`.

## Definition

For a real sequence `x[n]` of length `N`:

`
H[k] = Σ_{n=0}^{N-1} x[n] · cas(2πkn/N)
`

where `cas(θ) = cos(θ) + sin(θ)`.

## Properties

- **Real-to-real**: both input and output are real, saving storage vs. DFT
- **Self-inverse**: the inverse DHT is the same transform scaled by 1/N
- **Relation to DFT**: `H[k] = Re(X[k]) − Im(X[k])`

## API

```rust,ignore
use apollo_dht::{DhtPlan, dht_forward, dht_inverse};

let plan = DhtPlan::new(length)?;
let hartley = dht_forward(&signal, &plan)?;  // [N] -> [N] real
let original = dht_inverse(&hartley, &plan)?;
```

## Uses

DHT is useful in:
- **Image processing** — some filter operations are cheaper in the Hartley domain
- **Autocorrelation** — via the convolution theorem for real signals
- **Signal compression** — early JPEG-like codecs

DHT is not currently used in the primary Atlas signal processing pipelines,
but is available for research and exploration.
