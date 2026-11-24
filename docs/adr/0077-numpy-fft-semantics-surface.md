# ADR 0077: NumPy FFT semantics surface

- Status: Proposed
- Date: 2026-11-24; [major], released as `apollo-fft` 0.32.0; basis `b5e8b70b`
- Items: `APOLLO-CAP-BATCH-AXIS-ND`, `APOLLO-CAP-NORMALIZATION-MODES`,
  `APOLLO-CAP-FALLIBLE-API`, `APOLLO-CAP-KERNEL-SURFACE`,
  `APOLLO-MEM-CACHE-BOUNDS`, `APOLLO-COR-IRFFT-SPLIT-LENGTH`,
  `APOLLO-NUMPY-SEMANTICS-TOUR`, `APOLLO-DOC-NUMPY-MIGRATION` (the guide)

## Context

Five of the six `M-numpy-fft-semantics` exits need public shapes no record
fixes (`tests/frequency_axes.rs` passes at the basis):

- `Normalization` has one variant, `FftwCompatible`, and is exhaustive,
  serde-derived and the type of `BackendCapabilities::normalization`.
  [ADR 0067](0067-normalization-modes.md) (Accepted) renames and extends it
  and says "plans and `api` take the mode", naming no signature.
- Axis forms are eight asserting `FftPlan2D`/`FftPlan3D` methods
  (`{forward,inverse}_axis_complex{,_leto}_inplace`); Leto 0.52 views are
  rank-generic and `fftshift_axes` takes `axes: &[usize]`.
- None of the 78 `pub fn` in `api/`, the 49 `forward_*`/`inverse_*` methods of
  the 1-D, 2-D and 3-D plans or the 31 of `RealFftData`
  (`real_storage.rs:56-485`) returns a `Result`; inputs are checked by
  `Shape*::new(..).expect`, `assert_eq!` and `debug_assert_eq!`.
  [ADR 0072](0072-fallible-public-fft-api.md) (Proposed) makes them fallible.
- Nine plan methods carry a mode in their names (`inverse_*_unnorm_*`), called
  from apollo-nufft, apollo-qft, apollo-sht and apollo-fft's kernels
  (`kernel/auto_dispatch.rs:259`). `kernel::fft_forward`, `fft_inverse` and
  `fft_inverse_unnorm` (`auto_dispatch.rs:68-80`) are public through
  `pub mod kernel`; ADR 0067 slice 3 narrows them. The one retained-bytes
  measure, `batched::cache::retained_bytes_f64`, is `#[cfg(test)] pub(crate)`.

NumPy 2.5.2 (v2.5 manual "Normalization", read; runs this session):
`"backward"` scales the inverse by 1/n, `"ortho"` both by 1/sqrt(n),
`"forward"` the forward by 1/n; `fftn` repeats a repeated axis; an axis outside
`-rank..rank` raises `IndexError`; a zero transformed extent raises
`ValueError`. The Cargo Book (semver reference, read 2026-11-24): "only changes
in the left-most non-zero component are considered incompatible", so 0.31 to
0.32 breaks, as does a variant added without `non_exhaustive`; the checklist
and `gap_audit.md` now say [major] for 0.32.0, as the ladder and board do.

## Decision

1. **Mode.** `Normalization` becomes `#[non_exhaustive]` with `Backward`
   (`FftwCompatible` renamed, the old name deleted; its serde name changes, no
   persisted use found), `Orthonormal` and `Forward`, passed as `norm`. One
   `match` at the operation boundary selects the monomorphized `Direction`
   passes and at most one unfused scale pass, its factor computed once in `T`.
   With N the product of the transformed extents the forward scales by 1,
   1/sqrt(N), 1/N and the inverse by 1/N, 1/sqrt(N), 1; under `Backward` the
   plans run today's passes, fused 1/n included, so outputs stay bitwise. The
   axis-set functions (2) and every plan `forward_*`/`inverse_*` take the mode;
   the nine `_unnorm_` methods are deleted (callers pass `Forward`, QFT
   `Orthonormal`); the rank-named `api` functions keep `Backward` and call the
   same plan methods. ADR 0067's body narrows to this on acceptance.
2. **Axis sets.** `fft_axes<T, const N: usize>(field: ArrayViewMut<'_,
   Complex<T>, N>, axes: &[usize], norm: Normalization) -> ApolloResult<()>`
   and `ifft_axes`, in place over Leto views of any layout. `T` carries the
   bound of `fft_1d_complex_inplace` (`api/cfft.rs:70-73`), implemented for
   `f64` and `f32` alone (`scalar/impls.rs:89,357`); `F16` has `PlanScalar =
   f32` (`real_storage/compact.rs:10`), so it takes the real forms only. `&[k]`
   is `numpy.fft.fft(a, axis=k)`, a set is `fftn(a, axes=...)`, an empty set
   changes nothing, and a zero extent on a non-transformed axis returns `Ok`
   with no work (NumPy: empty result). Axes apply in the order given, the
   result order-independent up to rounding, which the oracle bounds. Per axis
   the C-order shape collapses to `[outer, n_a, inner]` and the existing 3-D pass
   runs `lane_over`; lane twiddles are the 3-D plan's (a cached power-of-two
   table, else `lane_over(None)`, which reads the plan-less caches ADR 0068
   slice 2 re-homes), so `fft_axes` adds no reader to that inventory. No kernel
   or N-D plan type is added; the eight plan axis methods are deleted.
   The Python layer maps negative axes to `axis + rank`. Real forms land in
   `APOLLO-CAP-PYTHON-SURFACE`; crop and pad (`n`, `s`) are out of scope.
3. **Fallibility.** ADR 0072 applies in place, with no `try_` family, to the 78
   `api` functions, the 31 `RealFftData` methods that execute a transform
   (`_via_split` probes return whether they ran) and the 32 plan execute
   methods left once (1) and (2) delete 17 of the 49; `RealFftData` takes no
   `norm` until the Python item gives real forms modes. `fft_axes` is fallible
   from birth. `ApolloError` becomes `#[non_exhaustive]`, no new variant.
   Preconditions map to: zero-length input or transformed extent,
   `Validation` naming the field (`n`, `nx`, `ny`, `nz`, `axes`) with value
   `0`, as `Shape*::new` does; an axis not below the rank, or repeated,
   `Validation` for `axes`, as `fftshift_axes` returns (NumPy repeats the
   transform; Apollo refuses); an `_into` output or `scratch`, a plan shape
   against the data, a static `N` against the runtime length,
   `ShapeMismatch { expected, actual }`; a buffer a route needs contiguous,
   `NonContiguous`; kernel-internal invariants stay `expect` calls.

   Kernels inside apollo-fft build the plan from the data's own length, so a
   mismatch is unreachable there: each public plan execute method is a
   release-checked validation over a `pub(crate)` core that the kernels call,
   revising ADR 0072's "propagate with `?`" (it would make the public
   `FftPrecision::fft_*` fallible). Python exception mapping is the Python
   item's (today `ValueError` from `err.to_string()`, `dht.rs:21`).
4. **Retained bytes.** `pub fn retained_plan_bytes() -> usize` beside
   `clear_plan_caches()` counts every table reachable from the plan and kernel
   caches once by address, over every scalar; the clear prunes dead `Weak`
   entries, so with no plan held the value returns to its post-clear baseline.
   The counting rule is in `APOLLO-MEM-CACHE-BOUNDS`; it presumes
   [ADR 0068](0068-bounded-plan-and-table-caches.md) slice 2.
5. **Release and consumers.** The first breaking slice to land
   (`APOLLO-CAP-BATCH-AXIS-ND`) sets `apollo-fft` 0.32.0 and moves the 18
   sibling `version = "0.31.0"` requirements (`apollo-python` is path-only) in
   the same commit; publication needs release authority. The overlay patches
   only the git source of `apollo-fft` (`.cargo/config.toml`) and the Cargo
   Book says the patched version "will affect whether the patch is used", so
   git-sourced consumers keep 0.31.0 until each migrates once, against the
   finished surface. Reverse dependencies at the fetched defaults:
   - direct `api` calls: kwavers 90 in 42 files (plus 14 plan calls, 2 axis
     calls, 4 `RealFftData` calls, `FftwCompatible` once), CFDrs 8 in 3,
     coeus 4 in 1, helios 3 in 1;
   - requirement edge: ritk, with 10 `kernel::fft_*` calls in 5 files;
   - transitive: asclepius, through the `coeus-autograd` lock;
   - unaffected: the other 21 members (`consus` names only
     `apollo-dctdst-core`); none has an `apollo-fft`/`-nufft`/`-sht` edge.

   Migration order: siblings, coeus, ritk and asclepius, then helios, kwavers
   and CFDrs. kwavers re-exports `Normalization` and `FftPlan1D/2D/3D`
   (`kwavers-math/src/fft/mod.rs:18-21`) and exposes
   `FourierBackend::normalization()`, so its published crates break too.
   Siblings returning `ApolloResult` (apollo-nufft, apollo-sft) or implementing
   `GpuTransformPlanner` (apollo-sht) version by `cargo-semver-checks` class,
   and the CFDrs and ritk requirements follow.

## Rejected alternatives

- **Additive 0.32.0**: a second mode type and fallible siblings beside the
  panicking forms; it contradicts Accepted ADR 0067 and moves the break on.
- A type-parameter mode (a branch per lane); `norm` on the rank-named
  functions; an `FftPlanNd`; keeping the plan axis methods;
  `retained_plan_bytes::<T>()` (several tables have no scalar); bumping the
  manifest in the last slice (overlay consumers meet each break as it lands).

## Consequences

- 0.32.0 breaks the renamed variant, `#[non_exhaustive]` on `Normalization`
  and `ApolloError`, every `api`, plan and `RealFftData` execute signature, and
  the deleted `_unnorm_`, axis and `kernel::fft_*` items; `BREAKING CHANGE:`
  footers and the guide carry the migration. `BackendCapabilities::
  normalization` stays `Backward` until ADR 0071 lands.
- The rank-named family overlaps `fft_axes` at ranks 1-3; the stable-suite
  milestone decides whether 1.0 keeps both and files the break typing
  `ApolloError`'s `String` payloads. ADR 0076's interop "axis views" row is
  delivered by Leto views and carries a revision note.

## Overturning evidence

A user veto of the break: the additive redirect. Axis plans needed outside the
bounded cache: an `FftPlanNd`. A scale pass above the `Backward` drift band:
fuse it per route. Parity cases needing crop and pad: widen (2).
