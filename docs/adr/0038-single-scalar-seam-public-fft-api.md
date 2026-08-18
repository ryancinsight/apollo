# ADR 0038: One scalar seam for the public FFT API

- **Status:** Accepted
- **Date:** 2026-08-17
- **Class:** [major] [arch]

## Context

`crates/apollo-fft/src/api` exposed 140 public functions. 138 of them formed
69 pairs in which a concrete `f64` function and a generic `_typed` function
computed the same result:

- 59 pairs where the concrete function's whole body was
  `X_typed::<f64>(args)`.
- 10 pairs where the concrete body was the generic body with `T` written out
  as `f64` — copy-paste rather than delegation.

The remaining two functions, `fft_1d_slice_typed` and `ifft_1d_slice_typed`,
had no concrete sibling.

The pair set was verified mechanically rather than by inspection: each
concrete body was normalised (`<f64 as Trait>::m` and `f64::m` to `T::m`,
`::<f64>` to `::<T>`, `Complex64` to `Complex<T::PlanScalar>`, own name to
twin name, rustfmt-only whitespace and trailing commas removed) and compared
token-for-token against its generic twin. All 69 matched exactly; none was a
near-twin hiding a behavioural difference.

The duplication was not caused by a missing scalar seam. Apollo already has
one, correctly layered:

```
eunomia::RealField          (sealed; impls: f32, f64)
  └─ WinogradScalar
       └─ ShortWinogradScalar
            └─ CompositeCache, BluesteinStore
                 └─ MixedRadixScalar    (sealed; impls: f32, f64)

RealFftData                 (impls: f16, f32, f64)
  └─ PlanCacheProvider      (impls: f16, f32, f64;  PlanScalar: MixedRadixScalar)
```

Only four bounds appear anywhere in `api/`: `MixedRadixScalar`,
`PlanCacheProvider`, `RealFftData`, and the `Complex<T::PlanScalar>:
PlanScratch` companion clause. The `_typed` suffix therefore encoded no
variation dimension — it marked which of two spellings of one operation the
caller reached. That is the naming prohibition's "additive marker": a name
element distinguishing nothing in the contract.

## Decision

The public FFT API exposes exactly one entry point per operation, generic over
the scalar seam. Concretely:

1. **Complex transforms** bound on
   `T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>`.
2. **Real transforms** bound on `T: PlanCacheProvider`, with the companion
   clause `Complex<T::PlanScalar>: PlanScratch`.
3. No `_typed` spelling, no concrete-scalar sibling, and no re-export bridging
   the two. The generic function takes the concrete function's name, so
   call sites passing `Array1<Complex64>` continue to compile by inference.

`PlanCacheProvider` is the seam named in (2) because `PlanCacheProvider:
RealFftData` already. The 23 sites written `T: RealFftData + PlanCacheProvider`
(and the six adding `+ Copy`, likewise implied) restate a supertrait. That
redundancy is **recorded, not yet removed**: this change deletes duplicate
functions and does not rewrite the surviving bound lists, so the reduction to
the single bound is follow-up work under `APOLLO-COMPLEX-PAIR-SEAM-050B`. The
decision above fixes which trait is the seam; the spelling cleanup follows it.

### Rejected: introduce a new unifying scalar trait

The strongest alternative was a new `FftScalar` trait blanket-implemented over
the existing chain, giving one bound for both complex and real APIs. Rejected:
the complex and real APIs are not parameterised over the same thing. Complex
transforms are generic over the *arithmetic* scalar (`f32`, `f64` — what
`MixedRadixScalar` seals). Real transforms are generic over the *storage*
scalar (`f16`, `f32`, `f64`) and reach the arithmetic scalar indirectly
through `PlanScalar`. Collapsing them would either drop `f16` storage support
or force `f16` into an arithmetic seam that has no `f16` implementation. Two
bounds is the correct count; the defect was the duplicated spelling, not the
seam.

### Rejected: keep the `_typed` names and delete the concrete ones

This preserves the same function count reduction but leaves every call site
carrying a suffix that names no variation dimension, and breaks every existing
concrete call site for no gain. Taking the concrete name is source-compatible
wherever the scalar is inferable.

## Theorem / contract

For every collapsed pair `(C, G)` with `C` the deleted concrete function and
`G` the retained generic one, the retained definition is *the same code object*
that `C` already executed:

- Delegating pairs: `C(args) ≡ G::<f64>(args)` was `C`'s literal body, so
  every call `C(args)` is replaced by a call that was already made on its
  behalf. The emitted machine code for `f64` callers is `G`'s `f64`
  monomorphisation, which existed before this change.
- Copy-paste pairs: `C`'s body was proven token-identical to `G[T := f64]`, so
  `G`'s `f64` monomorphisation is the same instruction sequence the compiler
  already produced for `C`.

Equality is therefore **bitwise and by construction**, not approximate: no
tolerance is derivable or required, because no arithmetic changed. The
existing round-trip and Parseval oracles in `lib_tests/` and
`examples/book_parseval.rs` remain the behavioural gate, and they cover the
retained generic path both before and after — before as `G`, after as `G`
under its new name.

Instantiation count is unchanged. Both `f32` and `f64` monomorphisations of
each generic were already built (the generic functions were public and
exercised at both scalars); deleting a wrapper that called one of them removes
a symbol without adding a specialisation.

## Consequences

`apollo-fft` advances 0.26.0 → 0.27.0. The public surface of `api/` falls from
140 functions to 71 (69 deleted, 71 retained under their collapsed names);
`api/mod.rs` re-export lists are regenerated from the definitions rather than
hand-maintained, since the collapse mapped two former names onto one.

Two call-site classes break loudly rather than silently:

- **Static (const-generic) call sites.** `f::<8>(x)` no longer resolves,
  because Rust requires all-or-nothing explicit generic arguments — verified
  directly: `go::<8>` against `fn go<const N: usize, T>` is `E0107`, so no
  parameter ordering avoids this. Such sites become `f::<f64, 8>(x)`. Const
  parameters keep their existing position after `T`, matching the retained
  generic signatures.
- **The 3D inverse-real naming anomaly.** Apollo shipped
  `ifft_3d_array_into` bound to the *spectrum-scratch* kernel and
  `ifft_3d_array_into_scratch` bound to the *explicit-scratch* kernel — the
  inverse of the 1D and 2D convention, where `X_into` is explicit-scratch and
  `X_into_spectrum_scratch` consumes the spectrum. The collapse cannot
  preserve both spellings, and preserving the 3D reading would propagate the
  inconsistency into the merged name. 3D moves onto the 1D/2D convention:
  `ifft_3d_array_into` is now explicit-scratch (three arguments) and
  `ifft_3d_array_into_spectrum_scratch` consumes the spectrum. Because the
  arities differ (3 versus 2), every existing 3D caller fails to compile
  rather than silently changing kernel.

Out of scope, recorded for follow-up:

- `stockham/avx/{precise,reduced}` remains split. The two directories are not
  a scalar fork of one body: they implement different fused-stage sets (65
  versus 53 functions; `precise` alone has the `len32` fixed path and
  `stage_triple_groups_eight`, `reduced` alone has `stage_pair_quarter` and
  `stage_triple_quarter_groups_two`) over different vector geometries — f64
  packs 2 complex per YMM and deinterleaves with one `permute2f128`, f32 packs
  4 and needs a two-stage `unpacklo/hi_pd` + `permute2f128` transpose network.
  Merging them requires first introducing a SIMD-vector seam (an associated
  vector type with `deinterleave`/`cmul`/`store` operations per scalar), which
  is a separate `[arch]` decision with its own differential-verification
  burden. The directory names remain a naming-prohibition violation and are
  renamed to their domain concept under that item, not this one.
- The eight sealed two-implementation traits over the `Complex64`/`Complex32`
  pair (`KernelScalar`, `FftPrecision`, `PlanScratch`, `TwiddleOutput`,
  `TwiddleStore`, `NormalizeSlice`, `ScratchDispatch`, `StockhamKernel`) each
  re-declare the same type pair with a private `Sealed` module. That is the
  remaining seam fragmentation in apollo-fft; it is invisible from `api/` and
  is consolidated separately.
