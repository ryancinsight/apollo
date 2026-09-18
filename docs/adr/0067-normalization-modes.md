# 0067 — Normalization modes through a direction strategy

- Status: Accepted
- Date: 2026-09-18
- Revised 2026-09-18: slice 1 delivered. QFT and SHT already used the 1-D
  plans' unnormalized inverse, so the only workaround that moved is the 1-D
  NUFFT's normalize-then-multiply.
- Items: `backlog.md#apollo-cap-normalization-modes`
- Evidence: `output/apollo-capability-audit-2026-09-18.md` item 3

## Context

`apollo-fft` has one normalization convention: the forward transform is
unnormalized and the inverse divides by the transformed volume
(`Normalization::FftwCompatible`, the enum's only variant). An unnormalized
inverse exists on the 1-D plans only (`inverse_complex_*_unnorm_*`). The 2-D
and 3-D plans, and the free-function `api`, normalize unconditionally. Three
in-tree consumers work around the gap:
- QFT takes the unnormalized 1-D inverse and scales by `1/√N` itself.
- SHT reaches the 1-D unnormalized inverse.
- The 1-D NUFFT normalizes and then multiplies back by `M`.

NumPy and SciPy callers expect `norm ∈ {backward, ortho, forward}`.

In the 2-D and 3-D paths normalization happens per axis, inside the lane
transform. The lane executors (`plan/fft/lanes.rs`: `execute`, `contiguous`,
the four-step `transform`, `lane_over`) and every pass above them carry a
`const FORWARD: bool`, and the inverse's normalization is implied by
`FORWARD = false`. An unnormalized inverse therefore has no channel through
the passes.

## Decision

Replace the passes' `const FORWARD: bool` with a zero-sized direction
strategy, `D: Direction`, whose associated constants give the kernel
direction and whether the inverse normalizes.
- The strategies are `Forward`, `Inverse` (normalized) and
  `InverseUnnormalized`.
- The lanes select the kernel from `D::FORWARD` and `D::NORMALIZE`, so every
  pass monomorphizes per strategy with no runtime branch (static routing).

On top of that channel:
1. **Slice 1.** The 2-D and 3-D plans gain unnormalized inverses
   (`inverse_complex_unnorm_inplace` and the Leto form), matching the 1-D
   plans. The 1-D NUFFT's normalize-then-multiply moves to the 1-D
   unnormalized inverse.
2. **Slice 2.**
   - `Normalization` gains `Backward` (today's behavior; `FftwCompatible`
     is renamed to it), `Orthonormal` and `Forward`, and becomes
     `#[non_exhaustive]`.
   - Plans and `api` take the mode.
   - `Orthonormal` runs the unnormalized transforms and scales by
     `1/√N` once per call. `Forward` scales the forward by `1/N` and runs
     the unnormalized inverse.
   - The scale is one pass over the output, fused into the last lane pass
     where the route allows.
3. **Slice 3.** The kernel-internal `kernel::fft_*` free functions become
   `pub(crate)`.

## Alternatives

- **Normalize, then multiply back.** Rejected: this is the workaround the
  audit found in NUFFT. It costs a pass and rounds twice, and the round trip
  of `1/n_x · 1/n_y · N` is exact only for powers of two.
- **A second `const NORMALIZE: bool` beside `FORWARD` on every pass.**
  Rejected: it adds a meaningless combination (a normalized forward), and
  doubles the parameter list on 36 sites that would each forward both.
- **Runtime normalization flags.** Rejected: a branch per lane where the plan
  already routes at construction.

## Consequences

- Slice 1 is additive: new methods, same outputs for the existing ones, bit
  for bit, because `Inverse` selects the kernels `FORWARD = false` selected.
- Slice 2 changes a public enum and is breaking: a minor version bump before
  1.0, with a CHANGELOG migration note.
- GPU normalization (Hephaestus) is a non-goal here.

## Verification

- **Existing paths:** the forward and the normalized inverse are bit-identical
  to before, and the per-axis sequences agree with the whole transform.
- **Unnormalized inverse:** it equals the normalized inverse times the volume,
  within a derived bound, and exactly for powers of two.
- **Slice 2 modes:** for each mode and rank, the impulse and the constant give
  their analytic spectra; `Orthonormal` satisfies Parseval within a derived
  bound; every mode round-trips to the identity.
