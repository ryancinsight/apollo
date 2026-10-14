# Number Theoretic Transform

The Number Theoretic Transform (NTT) is a finite-field analogue of the FFT,
operating over ℤ/pℤ rather than ℂ. Apollo provides NTT through `apollo-ntt`.

## Purpose

NTT is used in:
- **Cryptographic applications** — polynomial multiplication in lattice cryptography
- **Error-correcting codes** — syndrome computation over GF(p)
- **Integer polynomial multiplication** — exact arithmetic without floating-point error

## API

`NttPlan::new` builds the twiddle tables for a power-of-two length over
the crate default prime; `with_modulus` takes an explicit prime `p` with
`p ≡ 1 (mod n)` and primitive root.

```rust
{{#include ../../crates/apollo-ntt/examples/book_ntt_roundtrip.rs}}
```

## Polynomial Multiplication

```rust
{{#include ../../crates/apollo-ntt/examples/book_ntt_polymul.rs}}
```

The flow is forward NTT of both inputs → pointwise multiply modulo the
prime → inverse NTT.

## Supported Moduli

`NttPlan::new` uses the crate default prime `DEFAULT_MODULUS` (998244353);
`with_modulus` takes an explicit prime with `modulus ≡ 1 (mod n)` and
primitive root, and `MERSENNE31` covers the circle-plan path.

## Relation to FFT

NTT and FFT share the same butterfly structure (Cooley-Tukey). Apollo reuses
Apollo's mixed-radix FFT plan infrastructure for NTT scheduling.
