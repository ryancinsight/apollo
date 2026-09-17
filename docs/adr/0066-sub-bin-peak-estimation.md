# 0066 — Sub-bin peak parameters from three bins and estimate-and-subtract

- Status: Accepted
- Date: 2026-09-15
- Revised 2026-09-16: the outer-tone floor was quoted as `5e-10 Hz`, which is
  the measurement, not the bound the cited expression gives (2.1e-9 and
  1.4e-9 Hz for the two tones), and the expression omitted the finite-length
  bias term the test asserts alongside it. The decision is unchanged.
- Revised 2026-09-17 (surface): the decision was implemented as `apollo_stft::estimate_peaks`
  (`backlog.md#apollo-peak-estimation-surface`). The spike's test-only
  candidates are deleted; the oracle scene is the surface's test and
  measured the Candan 2011 row below again (8.4e-10 and 7.8e-10 Hz, 6.0e-7
  Hz). The image-floor expression below is the first-order size of the image's
  effect, not an upper bound: review found estimates up to 15% above it near
  DC. The surface's tests instead bound every read by the triangle inequality
  on the exact image term, the residuals of the estimates actually subtracted,
  the scene's noise at the bins read, and rounding. The kernel is evaluated in
  the product form `e^{iπu'} sin(πu') e^{−iπw/N} / sin(πw/N)` on the argument
  reduced modulo `N`, since the quotient form cancels near integer
  arguments. The residual of Candan's correction for one complex exponential is
  `(N/π)(tan(πδ/N) − πδ/N)`, of which `|δ|³(π/N)²/3` is the leading term. A
  real tone also appears at its mirror bin `N − k`, so the surface refuses a
  peak set holding both.
- Revised 2026-09-17 (independent reference check, after the surface revision): the
  approximate correction
  maps a noise-free complex exponential at offset `δ` to
  `tan(πδ/N)/(π/N)`. At `δ = 1/2` this exceeds `1/2`, so the original
  rejection test discards an otherwise valid boundary estimate. The selected
  offset now inverts the tangent relation. The image solve rejects DC and
  even-length Nyquist by index: its previous `4ε(2k+3)` relative determinant
  threshold exceeds one at valid large-frame bins and rejects even an exact
  on-bin tone with zero image. See [reference checks](../../backlog.md#apollo-peak-independent-reference).
- Items: `backlog.md#apollo-spectral-peak-estimation-spike`,
  `backlog.md#apollo-peak-estimation-surface`,
  `backlog.md#apollo-peak-independent-reference`
- Evidence: `crates/apollo-stft/src/application/execution/kernel/peak/tests/`
  (`scene.rs` and `scene/reference.rs`: the oracle scene, every read of its
  three rounds checked against its derived bound, on the FFT spectrum and on
  direct DFT sums; `boundary.rs` and `tones.rs`: per-scalar cases),
  `output/apollo-base128/peak_estimation_2026-09-15.md` (the spike's run of
  every candidate); the candidates themselves are in git history at
  `crates/apollo-stft/src/application/execution/plan/stft/dimension_1d/tests/peak_estimation.rs`
  before the surface revision

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

## Independent production check, 2026-09-17

Against the surface revision (`8066cf25`), two new cases failed: an offset of
`-0.499999` at `N = 16`, phase `-2.4`, was rejected; and a unit on-bin cosine
at bin `1572864` of an `f32` frame of length `2097152` was rejected. The inverse-tangent closure and
index-based singularity check correct these cases without reducing the
accepted frame-length range.

Residual subtraction now follows strength order, with bin index breaking
equal-strength ties. Before this correction, permuting the three requested
bins changed the weak tone's complex coefficient in its last bits. All six
permutations now return identical corresponding estimates.

The scene replaces all nine bins read by the estimator with DFT sums of
the generated samples. It also runs a separately implemented geometric-series
quotient, image solve and subtraction recurrence. Every resolvable read is
checked over all three rounds. The deterministic error model includes the
tone's own image in the reference ratio, and adds noise, other-tone residuals,
sample/argument rounding and position readback error as perturbations. The
Cramér–Rao value is reported only as a lower-bound reference, not an upper
acceptance criterion.

The roundoff model assumes absolute sine/cosine error no greater than the
scalar epsilon; Rust does not guarantee this across platforms. Sample-angle
arithmetic and direct-DFT arithmetic have separate operation-count bounds
using `γ_m = m u/(1 − m u)`, `u = ε/2`. Integer reduction of each direct-DFT
phase keeps its magnitude below `2π`. The previous `2εΘ` sample allowance
omitted operations and was replaced from this derivation, not fitted to a
failure. These are conditional floating-point bounds and observed native
results, not a machine-checked proof of transcendental accuracy.

Measured mean absolute errors over eight seeds, three rounds, Windows x64,
Rust 1.97.0 (the run's output is `output/apollo-base128/peak-reference-tests.txt`;
a release-profile run reproduces the table at its displayed precision):

| Tone amplitude | FFT frequency / direct DFT (Hz) | FFT amplitude / direct DFT (V) | FFT phase / direct DFT (rad) |
|---|---|---|---|
| 1 V, lower | 8.01e-10 / 8.01e-10 | 1.08e-9 / 1.08e-9 | 2.51e-9 / 2.51e-9 |
| 1e-6 V, middle | 5.96e-7 / 5.96e-7 | 1.56e-11 / 1.56e-11 | 1.57e-5 / 1.57e-5 |
| 1 V, upper | 7.59e-10 / 7.59e-10 | 4.96e-10 / 4.96e-10 | 2.36e-9 / 2.36e-9 |

Agreement at the displayed precision is not bitwise FFT/DFT equivalence or
a statistical efficiency proof. Boundary tests cover `f32` and `f64`,
multiple lengths/phases, direct and mirrored reads, strict interior/exterior
offsets and exact half-bin real tones. At the exact boundary the image can
move the estimated offset to either side. Odd-frame adjacent conjugate peaks
remain unresolved by this isolated-tone ratio; no exact recovery is claimed
there. No latency or throughput measurement was made.

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
takes the middle tone from unresolvable to the errors measured below, because a tone
estimated to relative error `ε` leaves interference `ε` for the next round.
The DFT is linear, so the residual at any position is the raw spectrum minus
the subtracted tones' kernel terms — no re-transform per round.

**Historical spike measurements, mean absolute error over eight seeds, after
three rounds (not a measurement of the revised production implementation):**

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
image-limited rather than noise-limited: to first order the negative-frequency
image moves the offset by `δ(1 − δ²)(π/N)²/sin²(π(2k + δ)/N)` bins, plus Jacobsen
(3)'s finite-length bias `δ(π/N)²/3` which Candan's factor removes — 2.1e-9
and 1.4e-9 Hz for the two outer tones of this scene, against measurements of
5.7e-10 and 4.8e-10, a factor of three inside. Their own Cramér–Rao bound is
3.6e-13 Hz, three decades below, so the image and not the noise is what they
are against. The middle tone's Cramér–Rao lower bound on standard deviation is
3.6e-7 Hz and the historical mean absolute error is 5.2e-7 Hz. These are
different statistics: proximity does not establish a noise-limited estimator,
and a multiple of the lower bound is not an upper acceptance limit.

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

Apollo owns the three-bin complex-ratio estimator with the inverse-tangent
closure — `δ = atan(Re[(X_{k−1} − X_{k+1}) / (2X_k − X_{k−1} − X_{k+1})] ·
tan(π/N))/(π/N)` on the rectangular window — with amplitude and phase from the
image-corrected kernel solve, and estimate-and-subtract for multi-tone
scenes.

The inverse relation follows from the single-exponential identity
`r = tan(πδ/N)/tan(π/N)`. Since `|δ| ≤ 1/2` and `N ≥ 3`, its argument is
inside atan's principal branch. The earlier decision to omit this closure
based on the long-frame scene overlooked boundary rejection at short frame
lengths. The added atan removes that deterministic bias; it does not remove
real-tone image bias or noise. Candan, "Analysis and Further Improvement of Fine Resolution Frequency
Estimation Method From Three DFT Samples", IEEE Signal Processing Letters
20(9), 913-916, 2013 (doi:10.1109/LSP.2013.2273616), analyses this bias and
gives the bias-removed estimator. No performance improvement is claimed.

The public surface is an estimate-and-subtract entry over a spectrum and a
set of peak bins, returning frequency, amplitude and phase per peak, with
rejection when the estimated offset leaves the half-bin. Image bias and noise
can still move a true boundary tone across that test. A caller with one isolated tone
gets the single-pass path as the one-tone case.

## Consequences

- `apollo_stft::estimate_peaks` is the surface, generic over
  `eunomia::RealField` (`f32`, `f64`), with this file's oracle as its
  acceptance test; the scene runs in `f64` only, since its noise and middle
  tone lie below `f32`'s rounding of a 1 V spectrum at this length, and the
  per-scalar bounds are the kernel's unit tests.
- The rectangular window is the analysis window for peak estimation.
  `apollo-stft`'s Hann window keeps its own role; it is not the path to
  sub-bin accuracy.
- The image correction is a contract, not an optimization: a caller reading
  amplitude from `2|X_k|/|W̃(δ)|` is floored at `1/(2k)` relative, which the
  solve removes.
- No claim is made about the Prism method. Should Henry 2020 become
  retrievable, its windows drop into the same oracle as a further candidate.
