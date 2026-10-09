# ADR 0038: One scalar seam for the public FFT API

- **Status:** Accepted
- **Date:** 2026-08-17
- **Class:** [major] [arch]
- **Revised:** 2026-10-03 — compacted; `KernelScalar` deleted; pending-trait list restated from the code.

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

The pair set was verified mechanically: each concrete body was normalised
(`<f64 as Trait>::m` and `f64::m` to `T::m`, `::<f64>` to `::<T>`, `Complex64`
to `Complex<T::PlanScalar>`, own name to twin name, rustfmt-only whitespace
removed) and compared token-for-token against its generic twin. All 69 matched
exactly; none hid a behavioural difference.

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
PlanScratch` companion clause. The `_typed` suffix encoded no variation
dimension; it marked which of two spellings of one operation the caller
reached — the naming prohibition's "additive marker".

## Decision

The public FFT API exposes exactly one entry point per operation, generic over
the scalar seam:

1. **Complex transforms** bound on
   `T: MixedRadixScalar<Complex = Complex<T>> + PlanCacheProvider<PlanScalar = T>`.
2. **Real transforms** bound on `T: PlanCacheProvider`, with the companion
   clause `Complex<T::PlanScalar>: PlanScratch`.
3. No `_typed` spelling, no concrete-scalar sibling, and no re-export bridging
   the two. The generic function takes the concrete function's name, so
   call sites passing `Array1<Complex64>` continue to compile by inference.

`PlanCacheProvider` is the seam named in (2) because `PlanCacheProvider:
RealFftData` already. The 23 sites written `T: RealFftData + PlanCacheProvider`
(six adding `+ Copy`) restate a supertrait. This change deletes duplicate
functions and leaves the surviving bound lists alone; their reduction to the
single bound is follow-up work under `APOLLO-COMPLEX-PAIR-SEAM-050B`.

### Rejected: introduce a new unifying scalar trait

The strongest alternative was a new `FftScalar` trait blanket-implemented over
the existing chain, giving one bound for both complex and real APIs. Rejected:
complex transforms are generic over the *arithmetic* scalar (`f32`, `f64` —
what `MixedRadixScalar` seals), real transforms over the *storage* scalar
(`f16`, `f32`, `f64`), reaching the arithmetic scalar through `PlanScalar`.
Collapsing them would drop `f16` storage support or force `f16` into an
arithmetic seam with no `f16` implementation. Two bounds is the correct count;
the defect was the duplicated spelling, not the seam.

### Rejected: keep the `_typed` names and delete the concrete ones

This leaves every call site carrying a suffix that names no variation
dimension and breaks every existing concrete call site for no gain. Taking the
concrete name is source-compatible wherever the scalar is inferable.

## Theorem / contract

For every collapsed pair `(C, G)` with `C` the deleted concrete function and
`G` the retained generic one, the retained definition is *the same code object*
that `C` already executed:

- Delegating pairs: `C(args) ≡ G::<f64>(args)` was `C`'s literal body, so each
  call `C(args)` is replaced by a call already made on its behalf.
- Copy-paste pairs: `C`'s body was proven token-identical to `G[T := f64]`, so
  `G`'s `f64` monomorphisation is the instruction sequence the compiler already
  produced for `C`.

Equality is **bitwise and by construction**; no tolerance is derivable or
required, because no arithmetic changed. The round-trip and Parseval oracles in
`lib_tests/` and `examples/book_parseval.rs` remain the behavioural gate and
cover the retained generic path before and after. Instantiation count is
unchanged: both `f32` and `f64` monomorphisations of each generic were already
built, so deleting a wrapper removes a symbol without adding a specialisation.

## Consequences

`apollo-fft` advances 0.26.0 → 0.27.0. The public surface of `api/` falls from
140 functions to 71 (69 deleted, 71 retained under their collapsed names);
`api/mod.rs` re-export lists are regenerated from the definitions, since the
collapse mapped two former names onto one.

Two call-site classes break loudly rather than silently:

- **Static (const-generic) call sites.** `f::<8>(x)` no longer resolves,
  because Rust requires all-or-nothing explicit generic arguments (`go::<8>`
  against `fn go<const N: usize, T>` is `E0107`, so no parameter ordering
  avoids this). Such sites become `f::<f64, 8>(x)`; const parameters keep
  their position after `T`.
- **The 3D inverse-real naming anomaly.** Apollo shipped `ifft_3d_array_into`
  bound to the *spectrum-scratch* kernel and `ifft_3d_array_into_scratch` to
  the *explicit-scratch* kernel — the inverse of the 1D and 2D convention,
  where `X_into` is explicit-scratch and `X_into_spectrum_scratch` consumes the
  spectrum. The collapse cannot preserve both spellings, so 3D moves onto the
  1D/2D convention: `ifft_3d_array_into` is now explicit-scratch (three
  arguments) and `ifft_3d_array_into_spectrum_scratch` consumes the spectrum.
  The arities differ (3 versus 2), so every existing 3D caller fails to compile
  rather than silently changing kernel.

Out of scope, recorded for follow-up:

- `stockham/avx/{precise,reduced}` remains split. The directories are not a
  scalar fork of one body: they implement different fused-stage sets (65 versus
  53 functions; `precise` alone has the `len32` fixed path and
  `stage_triple_groups_eight`, `reduced` alone `stage_pair_quarter` and
  `stage_triple_quarter_groups_two`) over different vector geometries — f64
  packs 2 complex per YMM and deinterleaves with one `permute2f128`, f32 packs
  4 and needs a two-stage `unpacklo/hi_pd` + `permute2f128` network. Merging
  them first needs a SIMD-vector seam, a separate `[arch]` decision with its
  own differential-verification burden; the directory names (a naming-prohibition
  violation) are renamed under that item.
- Seam fragmentation behind `api/`, as read from the code on 2026-10-03.
  Unsealed: `FftPrecision` (public; `Complex64`, `Complex32`, `Complex<F16>`),
  `TwiddleOutput` (`pub(crate)`; the same three), and `StockhamKernel`
  (`pub(crate)`; implemented over the scalars `f64` and `f32`, not the complex
  pair). Sealed through a private supertrait, with `Complex64` and `Complex32`
  as the only implementors: `PlanScratch` (public), `ScratchDispatch` and
  `TwiddleStore` (`pub(crate)`; the latter through its cache and slot markers).
  `NormalizeSlice` (`pub(crate)`, sealed) is already one blanket impl over
  `Complex<F: FloatElement + Pod>`. The type-pair forks are consolidated
  separately.
