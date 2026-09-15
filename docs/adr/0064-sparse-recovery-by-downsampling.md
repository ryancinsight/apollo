# 0064 — Sparse recovery by downsampling and syndrome decoding

- Status: Accepted
- Date: 2026-09-15
- Item: `backlog.md#apollo-sft-sublinear-recovery`
- Evidence: `crates/apollo-sft/src/tests/downsampled.rs` (exact recovery,
  forced collisions, differential agreement), `crates/apollo-sft/benches/recovery.rs`
  (the scaling instrument), `output/apollo-base128/sft_recovery_2026-09-15.txt`

## Context

`apollo-sft` presents itself as the sparse Fourier transform owner and cites
the sublinear literature, but its one route is dense: an `O(N log N)` FFT
followed by an `O(N log K)` top-`K` heap. The plan already carries the
aliasing parameters a sublinear route needs — `bucket_count` and `trials` —
as unused domain data, and its module doc reserves the seam: "a later
sublinear isolation kernel can replace the infrastructure layer without
changing this public API".

## Options

1. **Hassanieh, Indyk, Katabi, Price (SODA 2012), "Simple and practical
   algorithm for sparse Fourier transform."** Random spectral permutation, a
   flat-window filter, `B = O(√(NK / log N))` buckets, `O(log N)` location and
   estimation loops with medians. Randomized, `O(log N · √(NK log N))`, and
   its correctness rests on the filter's leakage bounds — a large surface
   (window design, permutation inverses, median estimation) for a probabilistic
   guarantee.
2. **Hsieh, Lu, Pei (ICASSP 2013; arXiv:1407.8315), "Sparse fast Fourier
   transform by downsampling."** Downsample by `d = N / B` with `B = 4K`
   buckets and shifts `ℓ`: the DFT of `x[dn + ℓ]` aliases the spectrum as
   `d · x̂_{d,ℓ}[b] = Σ_j X[s_j] z_j^ℓ` over the tones `s_j ≡ b (mod B)`, with
   `z_j = e^{2πi s_j / N}`. The `2a` syndromes of a bucket holding `a` tones
   form a Prony system: the coefficients of `P(z) = z^a + c_{a-1} z^{a-1} + … +
   c_0` solve the Hankel system `m_{a+i} + Σ_j c_j m_{i+j} = 0`, its roots
   give the frequencies `s_j = N · arg(z_j) / 2π`, and the amplitudes solve the
   Vandermonde system. Deterministic given the collision counts; Theorem 1
   states recovery with probability `1 − B · (deK / (N(a_m + 1)))^{a_m + 1}`
   for uniformly distributed supports and cost `O(a_m · B log B)`, i.e.
   `O(K log K)` at `a_m = 4`.
3. **Pawar, Ramchandran (IEEE Trans. IT 2018), FFAST.** Coprime downsampling
   stages with singleton ratio tests and peeling over a bipartite graph.
   Order-one decoding only, but the stages need `N` with coprime factors;
   apollo's callers are power-of-two lengths.

## Decision

Option 2, as one plan kind selected through `RecoveryRoute::Downsampled` on
`SparseFftConfig` (`SparseFftPlan::downsampled(n, k)`), the dense top-`K`
route unchanged as `RecoveryRoute::DenseTopK` and the oracle. Enum dispatch
over the closed set is the plan-kind seam; the kernel lives in
`infrastructure/kernel/downsampled.rs`.

Two departures from the paper, both in the direction of a stricter contract:

- **The bucket count doubles on a failed round rather than halving.** For a
  power-of-two `N` every bucket count is a power of two and the counts nest,
  so two tones that collide at `B` collide at `B / 2` as well; halving `d`
  (the paper's step) can only lower the cost of the tones already resolved.
  Doubling `B` instead separates every collision by the time `B = N`, where
  each bucket holds one frequency and decoding is trivial. Recovery is
  therefore exact for any exactly `K`-sparse input in exact arithmetic — the
  guarantee is deterministic — and costs `O(K log K)` whenever no bucket at
  `B = 4K` holds more than `a_m = 4` tones (Theorem 1's event), doubling per
  failed round and degrading to the dense cost only for inputs that are not
  sparse at any count.
- **One syndrome more than the decoder consumes.** `2 a_m + 1` shifts are
  taken, so even the widest decode (`a = a_m`, which the `2 a_m` syndromes
  determine exactly) is checked against an equation it did not fit. A decode
  is accepted only when every root sits on the unit circle within the derived
  tolerance, snaps to a frequency in the bucket's residue class, the roots are
  distinct, and the amplitudes solved from the snapped roots reproduce every
  syndrome within the tolerance; a bucket failing all `a ≤ a_m` is carried to
  the next round.

Roots of the Prony polynomial (degree at most four) are found by the
Weierstrass (Durand–Kerner) simultaneous iteration with a Newton polish,
rather than the paper's closed forms: the closed forms for degrees three and
four are numerically fragile in complex arithmetic and the verification above
does not depend on how a candidate root was produced.

The candidate set the kernel returns passes through the plan's top-`K`
selection and threshold exactly as the dense spectrum does, so both routes
share one output contract: at most `K` coefficients, the largest by
magnitude, ties to the lower index.

## Tolerances

A syndrome carries the rounding of a length-`B` FFT of the shifted samples,
scaled by `d`: `d · c · log₂(B) · u · ‖x_{d,ℓ}‖₁` with `u` the unit roundoff
and `c` the same constant the apollo-fft oracles use (`16`). That bound is
the zero test for an empty bucket, the unit-circle band for a root, and,
times the tone count, the reconstruction tolerance for a decode. Frequencies
snap by rounding `N · arg(z) / 2π`, exact while the angular error stays under
`π / N`, which the reconstruction check enforces indirectly: a root far enough
off to snap wrongly does not reproduce the syndromes.

## Consequences

- `SparseFftPlan::forward` on a downsampled plan agrees with the dense route
  on every exactly `K`-sparse input: the same support, values within the
  derived bound. On inputs that are not sparse it returns the top-`K` of
  whatever the final round resolved, which at `B = N` is the full spectrum.
- The route requires a bucket count dividing `N` at every round, so
  `SparseFftPlan::downsampled` accepts only lengths whose factorisation admits
  it (a power of two times the starting count's cofactor); other lengths are a
  typed validation error naming the constraint, never a silent dense fallback.
- The failure of every round is a typed error (`ApolloError::Validation` on
  the signal, naming the unresolved bucket count), reachable only when the
  round cap `trials` is exhausted before `B` reaches `N`; the cap is
  `max(4, ⌊log₂ N⌋ + 1)`, which always suffices from `B = 4K`.
- The scaling claim is a local instrument, not a CI job: `benches/recovery.rs`
  runs the dense and downsampled routes at fixed `K` across `N` under a
  committed budget, and the pinned measurement is archived with this ADR.
