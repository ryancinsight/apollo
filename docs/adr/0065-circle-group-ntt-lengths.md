# 0065 — Power-of-two transform lengths over the circle group

- Status: Accepted
- Date: 2026-09-15
- Item: `backlog.md#apollo-ntt-circle-group-lengths`
- Evidence: `crates/apollo-ntt/src/application/execution/kernel/circle.rs` and
  `plan/ntt/circle.rs` tests (basis, round trip to `2¹⁶`, product modulo the
  vanishing polynomial), `crates/apollo-ntt/benches/circle.rs`,
  `output/apollo-base128/ntt_circle_2026-09-15.txt`

## Context

`NttPlan` is the radix-2 cyclic number-theoretic transform: it needs a
primitive `n`-th root of unity in `F_p`, hence `n | p − 1`. Mersenne
`2³¹ − 1` has `p − 1 = 2 · 3² · 7 · 11 · 31 · 151 · 331`, so it admits only
`n = 2`; its power-of-two structure sits on the other side, `p + 1 = 2³¹`.
Li and Xing (arXiv:2310.14462, Theorem 1.1) give an `O(n log n)` evaluation
map for `n | q + 1` through automorphism groups of the rational function
field; Haböck, Levit and Papini ("Circle STARKs", ePrint 2024/278) give its
practical form over the circle curve `x² + y² = 1`, whose `p + 1` rational
points form a cyclic group for `p ≡ 3 (mod 4)`.

## Decision

`apollo-ntt` gains one plan kind, `CircleNttPlan`, over the circle group,
beside the cyclic `NttPlan`. Its contract is stated as what it is — a
multipoint evaluation in a fixed non-monomial basis — not as a DFT:

- **Domain.** For `N = 2^n` and a prime supporting the order `n`
  (`2^{n+1} | p + 1`), the evaluation set is the standard position coset
  `D = Q · G_{n−1} ∪ Q⁻¹ · G_{n−1}` with `Q` of order `2^{n+1}` and
  `G_{n−1} = ⟨Q⁴⟩` (paper, Definition 2 and Proposition 1). `Q` is derived
  from a generator of the 2-Sylow subgroup found by lifting small `x` to the
  circle, deterministic per modulus. The plan exposes `D` in the layout the
  transforms use (`i < N/2` is `(x_i, y_i)`, `i + N/2` its conjugate).
- **Basis.** `b_k(x, y) = y^{k₀} v₁(x)^{k₁} ⋯ v_{n−1}(x)^{k_{n−1}}` with
  `v₁(x) = x` and `v_{j+1}(x) = v_j(2x² − 1)` (Definition 4). The forward
  transform maps values over `D` to coefficients in this basis; the inverse
  evaluates.
- **Algorithm.** The paper's circle FFT (Section 4.2): one split along `y`
  (`f₀ = (f(x, y) + f(x, −y)) / 2`, `f₁ = (f(x, y) − f(x, −y)) / 2y`) then
  `n − 1` splits along `x` under `π(x) = 2x² − 1`. The `x` values of each
  level are ordered so that `±x` sit half a block apart, which makes both
  directions in-place butterfly networks; the forward leaves the coefficients
  bit-reversed and one permutation restores the natural order. The halving is
  folded into the forward twiddles so that forward and inverse are exact
  inverses, not the paper's scaled basis (Remark 10).
- **Products.** The basis spans `L′_N = {p₀(x) + y p₁(x) : deg p_i ≤ N/2 − 1}`,
  one dimension short of the polynomials of total degree `N/2` (Lemma 6, the
  dimension gap), and the missing direction is the vanishing polynomial
  `v_n(x)` of `D`. The pointwise product of two functions over `D` therefore
  interpolates to their polynomial product modulo `v_n(x)` — the circle
  analogue of the cyclic transform's reduction modulo `x^N − 1` — and that
  is the convolution contract the plan states and the test pins.

## Alternatives

- Extending `NttPlan` to `F_{p²}` with a root of unity of order dividing
  `p² − 1 = (p − 1)(p + 1)`: a cyclic transform, but every element costs two
  field multiplications and the transform of a base-field input returns
  extension-field values; the circle FFT keeps everything in `F_p`.
- Li–Xing's Galois FFT directly: the same evaluation set up to isomorphism
  (Remark 4 of the circle paper), at double the multiplication count.

## Consequences

- Round trips are exact for every input (the tests pin `n ≤ 2¹⁶`); the
  forward of a unit coefficient vector is the basis polynomial evaluated over
  `D`; the pointwise product reduces modulo `v_n(x)` exactly, checked against
  a symbolic schoolbook product in canonical form.
- Cost `N · n` additions and at most `N · n` multiplications per direction,
  measured on the bench as `O(N log N)`.
- Non-goals stand: no accelerator path, no lengths beyond powers of two; a
  `B`-smooth generalisation would follow Li–Xing's larger automorphism
  groups.
