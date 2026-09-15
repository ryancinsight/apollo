# 0063 — The static-length real transforms take the split through the runtime kernel

- Status: Accepted
- Date: 2026-09-15
- Item: `backlog.md#apollo-real-static-split-bound`
- Evidence: `output/apollo-base128/static_split_probe_2026-09-15.txt` (the
  probe, `output/probe_static`), `crates/apollo-fft/tests/real_half_api/static.rs`

## Context

`fft_1d_array_static_into::<T, N>` and `ifft_1d_array_static_into::<T, N>`
bound `T: RealFftData` only and run the widening path at every length: the
real field is copied into complex storage and `StaticFftPlan1D::<_, N>`
transforms all `N` bins. Every dynamic real entry has taken the real split
since `#atlas-apollo-real-split-coverage` — `N` reals packed as `N/2` complex
samples, a half-length transform, the untangle — through the cached
half-length plan `PlanCacheProvider::get_1d_plan(N / 2)`. The static forms
have no plan provider in their bound, and their half length is `N / 2`, which
a const-generic argument cannot spell on the stable toolchain
(`generic_const_exprs`).

## Options

1. **Add `PlanCacheProvider` to the bound** and take the cached dynamic half
   plan. Rejected: a static form whose contract needs the plan cache is the
   dynamic form under another name, and the bound lands on every caller as a
   breaking change for information the compiler already holds.
2. **A second const parameter `HALF`** with a compile-time `HALF * 2 == N`
   check, the half transform `StaticFftPlan1D::<_, HALF>`. Rejected: stable
   and zero-cost, but every caller spells `N / 2` by hand at a public entry —
   API churn that `generic_const_exprs` would later make redundant.
3. **A literal size table**: `match N { 8 => half::<4>, 16 => half::<8>, … }`
   inside the default method. Rejected: the match instantiates the half
   transform at every listed size for every `(T, N)` a caller uses — the
   instantiation fan-out the size gate forbids — for a route the runtime
   kernel already provides.
4. **The runtime kernel.** The half transform runs through
   `dispatch_inplace::<F, INVERSE, NORMALIZE>(packed, None)` — the
   plan-free entry every non-power-of-two lane of the 2-D and 3-D plans
   already takes — with its twiddles and radices from the process caches and
   the split twiddles evaluated on the fly as the dynamic forms do. No bound
   changes, no new parameter; the static form keeps its zero-sized plan for
   lengths the split refuses.

## Decision

Option 4. `RealFftData::inverse_1d_static_into` takes the split wherever it
admits `N` and the arrays are contiguous, and `forward_1d_static_into` from
`N = 128` or at any non-power-of-two `N`, both through the runtime kernel;
the widening path stays for the lengths the split refuses, for the short
powers of two on the forward, and for strided storage. The item's change
class falls from [major] to [minor]: no public bound or signature moves, and
`cargo-semver-checks` sees only the behaviour behind unchanged surfaces.

What the route gives up is the const-`N` executor for the half: the plan-free
dispatch selects its algorithm from the length at run time, where
`StaticFftPlan1D` monomorphizes the selection. The probe measured that trade
against the widened static path (minimum per call over 200 samples, widened
over routed, two runs):

| N | forward | inverse |
|---|---|---|
| 4 | 0.44 | 1.37-1.54 |
| 8 | 0.45 | 1.14 |
| 16 | 1.04-1.06 | 1.01-1.06 |
| 20 | 1.08-1.14 | 1.53 |
| 32 | 0.86 | 1.43 |
| 64 | 0.95-1.25 | 1.33-1.37 |
| 128 | 1.02 | 1.36 |
| 256 | 1.35-1.36 | 1.51-1.54 |
| 1024 | 1.11-1.14 | 1.08-1.12 |
| 4096 | 1.03 | 1.13-1.21 |
| 65536 | 1.20-1.22 | 1.35-1.48 |

The inverse wins at every length: the widened path copies the whole
spectrum into scratch where the split copies half. The forward loses only
where the static plan's constant-length power-of-two kernels run — 4 and 8
are a handful of nanoseconds the split's untangle alone exceeds, and 32 is
the strongest of the small kernels — so it keeps them below 128 and takes
the split above, and at every non-power-of-two length, where the static
plan has no such kernel. The dynamic split, on the cached plan's executor,
stays ahead of the static route at the larger lengths (65536: 116 against
130 µs), the cost the toolchain constraint imposes.

## Consequences

- The static entries agree with the dynamic split forms within the derived
  bound at every admitted `N` (they take different half-length executors, so
  bitwise identity is not the contract), and with the widening path within
  the same bound at the lengths the split refuses.
- Neither form allocates on a warm process: the runtime kernel's caches are
  the same the dynamic plans warm.
- Revisit when `generic_const_exprs` stabilizes: option 2's half transform
  becomes spellable as `StaticFftPlan1D::<_, { N / 2 }>` without a second
  parameter, and the const-`N` executor returns to the half at no API cost.
