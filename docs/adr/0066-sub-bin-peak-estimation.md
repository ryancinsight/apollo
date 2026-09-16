# 0066 — Sub-bin peak parameters from three bins and estimate-and-subtract

- Status: Accepted
- Date: 2026-09-15
- Revised 2026-09-16: the outer-tone floor was quoted as `5e-10 Hz`, which is
  the measurement, not the bound the cited expression gives (2.1e-9 and
  1.4e-9 Hz for the two tones), and the expression omitted the finite-length
  bias term the test asserts alongside it. The decision is unchanged.
- Item: `backlog.md#apollo-spectral-peak-estimation-spike`
- Evidence:
  `crates/apollo-stft/src/application/execution/plan/stft/dimension_1d/tests/peak_estimation.rs`
  (the candidate implementations, the oracle scene, and the derived bounds
  asserted), `output/apollo-base128/peak_estimation_2026-09-15.md` (the run)

## Context

Apollo reads spectra but owns no estimator of a tone's frequency, amplitude
and phase below bin resolution: `apollo-stft` supplies a Hann analysis window
and nothing that reads a peak. A caller wanting the parameters of a tone
between bins today writes its own interpolation.

The spike asked which published estimator Apollo should own. Its oracle is
the scene of Henry, Science Talks 4 (2022), Figs. 6-24: 48 kHz, 48 000
samples (a 1 Hz bin), three tones near 8950 Hz about 50 Hz apart, the outer
two at 1 V and the middle at 1e-6 V, white noise at 1e-10 V, random phases —
eight seeds, mean absolute errors.

## Findings

**Candidates.** Five resolved estimators of the fractional offset `δ` from
the peak bin, each implemented from its paper: Jacobsen–Kootsookos (IEEE SP
Mag 24(3), 2007) equations (3) complex-ratio, (4) parabolic-on-magnitudes
with `P = 1.36`, and (5) windowed with `Q = 0.55`; Candan (IEEE SPL 18(6),
2011), which is (3) scaled by `tan(π/N)/(π/N)`, and its 2013 closure
`arctan(δ π/N) N/π`; and Aboutanios–Mulgrew (IEEE TSP 53(4), 2005) Table I
Alg1, the DFT evaluated at `k + δ̂ ± ½` with
`δ̂ ← δ̂ + ½ Re[(X₊ + X₋)/(X₊ − X₋)]`, two iterations.

**Amplitude and phase need the window kernel and its image.** A real tone is
a conjugate pair, so bin `k` holds `a W̃(δ) + ā W̃(−(2k + δ))` for
`a = (A/2) e^{iφ}`. Reading `A = 2|X_k|/|W̃(δ)|` drops the second term and
floors the amplitude error at its ratio — 2e-5 here. Solving the pair
`X_k`, `X̄_k` for `a` removes it exactly. The kernels are closed forms (the
rectangular `R(u) = (e^{2πiu} − 1)/(e^{2πiu/N} − 1)`, the symmetric Hann as
three shifted copies), checked against the direct sum to 1e-5 absolute out to
17 802 bins.

**The middle tone is unresolvable in one pass, by construction.** Through the
Hann window the outer tones leak 4.98e-2 V into the middle tone's bin against
its own 1.09e-2 V; through the rectangular window the ratio is worse. No
three-bin estimator reads a peak there: every direct estimate either leaves
the half-bin (rejected) or errs in amplitude by more than the whole tone.
This is the leakage floor the Prism paper's windows exist to lower.

**Estimate-and-subtract removes it instead.** Estimating strongest-first on
the residual of the other tones' synthesized contributions, three rounds,
takes the middle tone from unresolvable to noise-limited, because a tone
estimated to relative error `ε` leaves interference `ε` for the next round.
The DFT is linear, so the residual at any position is the raw spectrum minus
the subtracted tones' kernel terms — no re-transform per round.

**Measured, mean absolute error over eight seeds, after three rounds:**

| estimator | tone 1 (1 V) | tone 2 (1e-6 V) | tone 3 (1 V) |
|---|---|---|---|
| Jacobsen (3), rectangular | 5.7e-10 Hz, 7.7e-10 V, 1.8e-9 rad | 5.2e-7 Hz, 1.2e-11 V, 8.6e-6 rad | 4.8e-10 Hz, 3.1e-10 V, 1.5e-9 rad |
| Candan 2011, rectangular | 8.4e-10 Hz, 1.1e-9 V, 2.6e-9 rad | 6.0e-7 Hz, 1.6e-11 V, 1.7e-5 rad | 7.8e-10 Hz, 5.1e-10 V, 2.4e-9 rad |
| Candan 2013, rectangular | 8.0e-10 Hz, 1.1e-9 V, 2.5e-9 rad | 6.0e-7 Hz, 1.6e-11 V, 1.6e-5 rad | 7.6e-10 Hz, 5.0e-10 V, 2.4e-9 rad |
| Aboutanios–Mulgrew, rectangular | 9.3e-6 Hz, 1.3e-5 V, 2.9e-5 rad | unresolved in 1 of 8 seeds | 1.6e-6 Hz, 1.3e-6 V, 4.9e-6 rad |
| Jacobsen (5), Hann | 1.6e-2 Hz, 7.7e-3 V, 5.1e-2 rad | 2.5e-1 Hz, 1.4e-7 V, 7.8e-1 rad | 3.0e-2 Hz, 8.2e-3 V, 9.3e-2 rad |
| Jacobsen (4), Hann magnitudes | 1.0e-3 Hz, 5.1e-4 V, 3.3e-3 rad | 3.8e-3 Hz, 1.7e-8 V, 1.6e-2 rad | 3.1e-3 Hz, 9.2e-4 V, 9.6e-3 rad |

The three-bin rectangular family lands within a factor of two of each other
and one to three decades ahead of every alternative. Their outer tones are
image-limited rather than noise-limited: the negative-frequency image bounds
the offset error at `δ(1 − δ²)(π/N)²/sin²(π(2k + δ)/N)` bins, plus Jacobsen
(3)'s finite-length bias `δ(π/N)²/3` which Candan's factor removes — 2.1e-9
and 1.4e-9 Hz for the two outer tones of this scene, against measurements of
5.7e-10 and 4.8e-10, a factor of three inside. Their own Cramér–Rao bound is
3.6e-13 Hz, three decades below, so the image and not the noise is what they
are against. The middle tone is the other way round: its Cramér–Rao bound is
3.6e-7 Hz and the measurement is 5.2e-7.

**Aboutanios–Mulgrew is the weakest of the rectangular candidates here, not
the strongest.** Its 1.0147×ACRB result is for one complex exponential in
noise; this scene is interference-limited, and its readings at `±½` bins sit
where a neighbouring 1 V tone leaks hardest, so it carries that interference
rather than the `δρ` the three-bin ratio divides away. It also costs two
direct DFT evaluations per iteration against three existing bins.

**Windowing loses on both counts.** The Hann candidates are slower to
converge (the window's own leakage is what the peel must remove, and the
estimator's `P`/`Q` constants carry a documented bias of 1e-3 to 3e-2 Hz that
no number of rounds removes) and the window does not lower the floor enough
to resolve the middle tone directly.

**Prism FFT is excluded as unreproducible.** Henry's `1e-12`-order claim rests
on two Prism-derived windows defined in Henry, IEEE TIM 69 (2020). That
reference was pursued to its open-access post-print (the Coventry portal
record returns HTML, not the PDF) and the 2022 Science Talks article
describes the windows only as "Prism-network impulse responses" without their
construction. Implementing it would require inventing the window definition,
so the comparison is not available and no claim about it is recorded.

## Decision

Apollo owns the three-bin complex-ratio estimator with Candan's length
correction — `δ = Re[(X_{k−1} − X_{k+1}) / (2X_k − X_{k−1} − X_{k+1})] ·
tan(π/N)/(π/N)` on the rectangular window — with amplitude and phase from the
image-corrected kernel solve, and estimate-and-subtract for multi-tone
scenes.

Candan 2011 over Jacobsen (3) at identical cost: the correction factor is
derived rather than fitted, and removes the `δ(π/N)²/3` finite-length bias
that dominates Jacobsen (3) at short `N` (at `N = 48 000` the two are within
the image floor of each other, which is why they measure alike). Candan 2013's
closure adds an `arctan` and measures no better; it is not carried.

The public surface is an estimate-and-subtract entry over a spectrum and a
set of peak bins, returning frequency, amplitude and phase per peak, with
rejection when an offset leaves the half-bin. A caller with one isolated tone
gets the single-pass path as the one-tone case.

## Consequences

- The estimator is test-only today. The DoR item
  `backlog.md#apollo-peak-estimation-surface` carries the public surface,
  generic over `T: Scalar`, with this file's oracle as its acceptance test.
- The rectangular window is the analysis window for peak estimation.
  `apollo-stft`'s Hann window keeps its own role; it is not the path to
  sub-bin accuracy.
- The image correction is a contract, not an optimization: a caller reading
  amplitude from `2|X_k|/|W̃(δ)|` is floored at `1/(2k)` relative, which the
  solve removes.
- No claim is made about the Prism method. Should Henry 2020 become
  retrievable, its windows drop into the same oracle as a further candidate.
