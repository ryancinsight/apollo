# Number Theoretic Transform

The Number Theoretic Transform (NTT) is a finite-field analogue of the FFT,
operating over ℤ/pℤ rather than ℂ. Apollo provides NTT through `apollo-ntt`.

## Purpose

NTT is used in:
- **Cryptographic applications** — polynomial multiplication in lattice cryptography
- **Error-correcting codes** — syndrome computation over GF(p)
- **Integer polynomial multiplication** — exact arithmetic without floating-point error

## API

```rust,ignore
use apollo_ntt::{NttPlan, ntt_forward, ntt_inverse};

// Choose a prime p with p ≡ 1 (mod 2^k) so 2^k-th roots of unity exist
let plan = NttPlan::new(length, modulus)?;

// Forward NTT: [u64] -> [u64] (mod p)
let transformed = ntt_forward(&input, &plan)?;

// Inverse NTT
let original = ntt_inverse(&transformed, &plan)?;
```

## Polynomial Multiplication

```rust,ignore
let product = apollo_ntt::polymul_ntt(&poly_a, &poly_b, modulus)?;
```

Internally: forward NTT of both inputs → pointwise multiply → inverse NTT.

## Supported Moduli

Apollo uses NIST-recommended 64-bit NTT-friendly primes with sufficient
bit-width for typical polynomial multiplication without intermediate overflow.

## Relation to FFT

NTT and FFT share the same butterfly structure (Cooley-Tukey). Apollo reuses
Apollo's mixed-radix FFT plan infrastructure for NTT scheduling.
