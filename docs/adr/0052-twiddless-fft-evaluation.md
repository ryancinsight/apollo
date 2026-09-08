# ADR 0052: Twiddless FFT evaluation

- **Status:** Rejected
- **Date:** 2026-09-08
- **Class:** [patch]
- **Item:** [APOLLO-TWIDDLESS-FFT-EVALUATION](../../backlog.md#apollo-twiddless-fft-evaluation)

## Context

Queiroz, "Fast compressed-domain N-point discrete Fourier transform: the
'twiddless' FFT algorithm" (arXiv:2505.23718v2, 2025), Algorithm 1, computes
an `N`-point DFT for `N = c · 2^k` by two rectangular-index-coefficient
compressions per level, `x̂e[n] = x[n] + x[n + N/2]` and
`x̂o[n] = (x[n] − x[n + N/2]) · exp(−2πi·n/N)`, recursion on both halves, and
the interleave `X[2c] = X̂e[c]`, `X[2c + 1] = X̂o[c]`, with a direct DFT base
case at `N ≤ 5`. The paper states `T(N) = 2T(N/2) + 7N − 4 ≈ 7N·log₂N` real
operations against `5N·log₂N` for radix-2 Cooley–Tukey, reports no
measurements and defers numerical analysis, and claims three structural
advantages: no butterfly, a combination step that is only index reordering,
and `c · 2^k` lengths without zero padding.

The request is to test and measure whether this algorithm can match or beat
Apollo's twiddle-and-butterfly routes.

## Analysis

The two compressions are the Gentleman–Sande decimation-in-frequency
butterfly (Gentleman and Sande 1966; Van Loan 1992, section 1.3). The
modulation `exp(−2πi·n/N)` on the odd half is the twiddle factor `W_N^n`;
the paper's `3N` real operations per level are that multiplication. Writing
both compressed halves contiguously is the in-place DIF stage's layout, and
recursing on both halves leaves the `2^k` leaf blocks in bit-reversed
residue order, so the index reordering of lines 12 and 14 is the DIF
bit-reversal gather. Apollo already runs this stage set in production on the
planar four-step's second axis (`components::batched::dif`). The `c · 2^k`
claim reduces to a direct `c`-point leaf under `k` radix-2 stages, which
every mixed-radix planner (including Apollo's) already provides with
`c`-point codelets.

No new operation, layout or accuracy property is therefore available to
adopt. The remaining question is empirical: does the out-of-place,
depth-first schedule the paper prescribes move data better than the
butterfly's in-place schedule, and does either approach Apollo's radix-4/8
Stockham and codelet routes?

## Instrument

`kernel/twiddless` (compiled under the `benchmark_kernels` boundary, never
routed) carries one arithmetic body and three sealed schedules:

- `CompressedHalves`: Algorithm 1 as published, depth first, out of place
  between the output and one scratch buffer of length `N`;
- `ButterflyRecursive`: the same traversal with the butterfly in place;
- `ButterflyIterative`: the classic breadth-first stage loop.

Per element the floating-point operation sequence is identical across
schedules, so a bitwise-equality test is the executed proof of the
isomorphism, and a timing difference between arms is buffer traffic alone.
Twiddles come from the crate's single twiddle authority (direct evaluation,
never a recurrence). Leaves 1, 2 and 4 fold their exact unit twiddles; 3 and
5 evaluate the definition. Admitted lengths are exactly the published base
case's: odd part 1, 3 or 5.

Verification (`twiddless/tests`): bitwise agreement of the three schedules at
every admitted length to `5 · 2^12`; forward error against a Dot2-compensated
direct oracle within the Higham Theorem 24.2 bound `8u·(t + 2)·‖X‖₂` plus the
oracle's derived `(4√N + 2)u·‖X‖₂`, both precisions, lengths to `5 · 2^8`;
distance to Apollo's production plan within the summed bound to `5 · 2^12`;
exact impulse spectrum and tone spectrum within bound; Parseval's identity
under proptest at random admitted lengths; length admission and rejection
with the reported odd part.

Measurement (`benches/twiddless_comparison`): clone-inclusive forward
transform, five arms (three schedules, Apollo's plan, RustFFT), `f64` and
`f32`, at `16, 64, 256, 1024, 4096, 16384, 65536` and the paper's `96, 640,
5120, 12288`, 20 ms warm-up and 80 ms measurement per case under
`apollo-bench`'s 100-sample median contract; budget 30 s, enforced.

Prediction, recorded before measurement: the published schedule is not
faster than its in-place isomorph at any length, and both trail Apollo's
routes at every power of two by the ratio of `7N·log₂N` scalar operations
to the vectorized radix-4/8 work. Stop condition: family-wise disjoint
intervals in the direction of the prediction at every power of two reject
the adoption question; an inversion at any length is a finding to attribute
before this record closes.

## Result and decision

Reject the twiddless FFT as a production candidate; retain the instrument
and its tests as the executed record.

Gate on the exact tree (origin/main `f9633c1c` plus this change): all-target
Clippy with the bench feature, the seven instrument tests, warning-denied
rustdoc with and without the feature, the crate doctest and the CI-style
bench smoke pass. One measurement run on logical processor 1 (performance
class) of the Intel Core Ultra 9 285K completed in 11.49 s of its 30 s
budget; evidence is `output/apollo-twiddless/measurement-run1.csv` with
`manifest.json` under Atlas retention. Medians in microseconds, 100 samples
per case, and the family-wise interval verdict of the published schedule
against Apollo's plan:

| N | f64 twiddless | f64 in-place | f64 Apollo | f64 RustFFT | verdict | f32 twiddless | f32 Apollo | verdict |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.037 | 0.033 | 0.009 | 0.011 | slower | 0.034 | 0.005 | slower |
| 64 | 0.167 | 0.151 | 0.048 | 0.046 | slower | 0.161 | 0.031 | slower |
| 256 | 0.803 | 0.735 | 0.246 | 0.204 | slower | 0.806 | 0.151 | slower |
| 1024 | 4.22 | 4.02 | 1.78 | 1.27 | slower | 3.77 | 0.850 | slower |
| 4096 | 26.0 | 23.9 | 8.86 | 8.06 | slower | 18.6 | 4.25 | slower |
| 16384 | 155 | 138 | 52.3 | 35.7 | slower | 174 | 24.8 | slower |
| 65536 | 799 | 771 | 1556 | 348 | faster | 631 | 885 | overlap |
| 96 | 0.620 | 0.580 | 0.130 | 0.094 | slower | 0.572 | 0.076 | slower |
| 640 | 5.63 | 5.47 | 0.900 | 0.737 | slower | 5.60 | 0.526 | slower |
| 5120 | 62.9 | 58.9 | 11.9 | 10.3 | slower | 50.3 | 7.21 | slower |
| 12288 | 145 | 125 | 37.1 | 27.2 | slower | 118 | 20.3 | slower |

Three schedules agree bitwise at every admitted length, so the arms differ
in data movement alone. The published out-of-place schedule is slower than
its in-place isomorph by 3 to 14% in `f64` with disjoint intervals at every
length below 65536, and never faster with support at any length; the
breadth-first stage loop is the fastest of the three below 16384. The
`c · 2^k` lengths the paper singles out are where the gap to Apollo is
widest (4.8 to 10.7 times), because Apollo's mixed-radix codelets serve
those lengths natively. No structural advantage survives measurement: the
compression is the butterfly, the modulation is the twiddle, the index
reordering is the bit-reversal gather.

The one inversion is not a twiddless property. At N = 65536 Apollo's
generic four-step route measures 1556 µs (`f64`, interval 1369 to 1741) and
885 µs (`f32`) against RustFFT's 348 µs and 90 µs, and a scalar radix-2
kernel with one scratch buffer beats it in `f64` with disjoint intervals.
That is filed as
[APOLLO-N65536-FOUR-STEP-SCALAR-LOSS](../../backlog.md#apollo-n65536-four-step-scalar-loss);
the wide intervals at that length mark it as a lead to re-measure under the
census supervisor, not a settled ratio.

Limits: one run on one host with concurrent load not captured (peer cargo
builds were excluded by the shared build-directory lock for the run's
duration); no cross-machine, inverse or multidimensional claim; the
instrument is scalar, so the comparison bounds the algorithm's structure,
not a vectorized implementation of it, which would inherit the same
operation count.

## Revision

2026-09-08: changed Proposed to Rejected after the measurement above.
