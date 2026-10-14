# Fast Walsh-Hadamard Transform

The Fast Walsh-Hadamard Transform (FWHT) expands a signal in Walsh
functions — square waves taking only the values `+1` and `−1`. Apollo
provides FWHT through `apollo-fwht`, including the native `f32` path.

## Definition

The Hadamard matrices recurse from `H_0 = [1]`:

`
H_{k+1} = [ H_k   H_k ]
          [ H_k  −H_k ]
`

so the length-`N = 2^k` transform is the matrix-vector product `H_k · x`,
computed in `N log N` additions and subtractions with no twiddle factors.

## Properties

- **Self-inverse up to scale**: applying the transform twice returns `N·x`,
  so the inverse is the forward transform scaled by `1/N`
- **Additions only**: every butterfly is one addition and one subtraction,
  which makes small transforms bit-exact
- **Power-of-two lengths**: plans reject any `n` that is not a power of two

## API

```rust
{{#include ../../crates/apollo-fwht/examples/book_fwht_roundtrip.rs}}
```

## Uses

FWHT is useful in:
- **Spread-spectrum codes** — Walsh codes separate CDMA channels
- **Transform benchmarking** — the cheapest exact transform, a lower bound
  for fancier kernels
- **Signal compression** — early sequency-ordered codecs
