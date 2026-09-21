# 0069 — One window for both STFT passes

- Status: Accepted
- Date: 2026-09-21
- Items: `backlog.md#apollo-cap-stft-window-inverse`
- Evidence: `output/apollo-capability-audit-2026-09-18.md` item 5

## Context

`StftPlan` analyzed and synthesized with different windows. `forward` applied
Hann; `forward_with_window` applied a caller's window; `inverse` always
synthesized with Hann. A caller who analyzed with Hamming and inverted got a
reconstruction with 4.1e-2 relative error, and nothing in the API or the
types said so — the two passes named the same plan, and the plan held no
window to disagree about.

The inverse also wrote zeros where the accumulated `Σ w²` was zero. That is
not a reconstruction: it is a silent substitution at exactly the samples the
window does not cover, and a symmetric Hann at `hop = frame` produces it at
both ends of every frame.

## Decision

The plan owns one window, and both passes use it.

- `StftPlan::new` keeps Hann. `with_window` takes a `Window` family (Hann,
  Hamming, Blackman, Tukey(α)); `with_window_values` takes caller values,
  validated for length and finiteness.
- `forward_with_window` is removed rather than deprecated. Its single in-tree
  caller (the apollo-validation STFT fixture) moves to `with_window_values`
  in the same change. A window that only one pass knows about is the defect
  this record exists to close, so the parameter does not survive in any form.
- Window values are evaluated at the nearer end (`x = min(i, n-1-i) / span`),
  so `w[i]` and `w[n-1-i]` are one evaluation and symmetry is bitwise, not
  approximate. Tukey's taper compares `edge < alpha / 2` rather than scaling
  `alpha`, so a subnormal α cannot underflow into "no taper".
- Where the overlap-add weight is at most `ε` times the largest, the inverse
  returns `StftError::WindowNotOverlapAdd` instead of dividing. The error
  enum gains that variant and `InvalidWindowParameter`, and becomes
  `#[non_exhaustive]`.

### Why the floor is `ε`, and what it does not mean

A frame's transform carries error relative to that frame's own largest
magnitude, `|w|max |x|max`, not relative to the sample being recovered. A
sample covered only by small window values therefore collects absolute error
of order `γ |w|max |x|max w_small` and is divided by a weight of order
`w_small²`, leaving a relative error bounded by `γ √(largest / weight)` with
`γ = 16 ⌈log₂ N⌉ ε`.

The amplification is the square root of the weight ratio, not its reciprocal.
The error reaches the sample's own magnitude only near
`weight / largest ≈ γ²` — `1.1e-28` at `N = 8`, `1.3e-27` at `N = 1024` —
which is twelve orders below `ε`. The floor is therefore a conditioning
choice, not the point of failure: at `ε largest` the bound is `γ / √ε`, about
`7e-7` at `N = 8` and `2.4e-6` at `N = 1024`, so an accepted plan still
reconstructs to roughly six significant digits, while below it the analysis
stops promising even half the mantissa.

This is recorded because the first draft of the check justified `ε` as the
point where "the division amplifies rounding past the sample's own
magnitude". That is false by about eight orders, and an independent review
measured it: a window one step inside the floor reconstructs with 4.3e-9
relative error, and the neighbouring refused configuration reconstructs just
as well. The threshold did not change; the reason for it did, from an
invented one to a derived one.

## Alternatives

- **Keep `forward_with_window` and have `inverse` take a window argument.**
  The two passes could then still disagree — the defect is that the window is
  a call-site parameter rather than plan state, and passing it twice preserves
  exactly that. Rejected.
- **Deprecate `forward_with_window` for one release.** The compatibility-shim
  prohibition applies: the branch is the isolation mechanism, the single
  caller migrates in this change, and a deprecated method that applies a
  window the inverse will not use keeps the wrong reconstruction reachable.
  Rejected.
- **Write zeros where the weight is zero, as before.** It returns a signal
  that is not the input and cannot be distinguished from one that is.
  Rejected in favour of a typed refusal.
- **Set the floor at the `γ²` crossing.** It is the true failure point, and
  it would accept every configuration this one accepts plus twelve orders
  more. Rejected for now: those configurations reconstruct with between six
  significant digits and none, and no caller has asked for that range. The
  crossing is recorded above so the choice can be revisited against a stated
  tolerance rather than re-derived.

## Consequences

Breaking, and the CHANGELOG carries the migration: `forward_with_window(&x,
&w)` becomes `with_window_values(frame, hop, w)` plus `forward(&x)`. A plan
built with a non-Hann window and inverted on the GPU is not a configuration
the GPU can detect — its payload carries only frame and hop lengths, so it
synthesizes with Hann. That pairing is the caller's error; the CPU plan is
the single source of the window, and ADR 0008's and ADR 0014's statement of
the WOLA identity narrows accordingly, from "wherever the denominator is
non-zero" to "wherever the denominator exceeds `ε` times the largest".

## Verification

- `every_window_reconstructs_what_it_analyzed` round-trips all four families,
  a ramp and a `1e6`-scaled Hann at two hops, asserting a bound derived from
  the frame count, the window's largest value and `γ` — scale-invariant, as
  the `1e6` case pins.
- `a_plan_at_the_floor_still_holds_six_digits` places the weight ratio one
  step inside the floor and asserts the `γ √(largest / weight)` bound, so the
  paragraph above is falsifiable rather than asserted.
- `every_window_is_symmetric` asserts bitwise equality, not a tolerance;
  `tukey_at_subnormal_alpha_tapers` fails against the `edge >= alpha / 2`
  form.
- `weights_follow_the_hop` and `uncovered_signal_ends_refuse_the_inverse`
  cover the symmetric-Hann-at-`hop = frame` refusal and the end-of-signal
  case; `stft_wgpu_inverse_refuses_a_hop_without_overlap` covers the GPU
  inverse against a real device.
