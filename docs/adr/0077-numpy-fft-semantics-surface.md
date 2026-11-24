# ADR 0077: NumPy FFT semantics surface

- Status: Proposed
- Date: 2026-11-24; [major], released as `apollo-fft` 0.32.0; basis `b5e8b70b`
- Items: `APOLLO-CAP-BATCH-AXIS-ND`, `APOLLO-CAP-NORMALIZATION-MODES`,
  `APOLLO-CAP-FALLIBLE-API`, `APOLLO-MEM-CACHE-BOUNDS`,
  `APOLLO-NUMPY-SEMANTICS-TOUR`

## Context

Five of the six `M-numpy-fft-semantics` exits need public shapes no record
fixes (the frequency-axes test is in source):

- `Normalization` has one variant, `FftwCompatible`, and is exhaustive,
  serde-derived and the type of `BackendCapabilities::normalization`.
  [ADR 0067](0067-normalization-modes.md) (Accepted) renames and extends it
  and says "plans and `api` take the mode", naming no signature.
- Axis forms are eight asserting `FftPlan2D`/`FftPlan3D` methods
  (`{forward,inverse}_axis_complex{,_leto}_inplace`, `axis < 3`). Leto 0.52
  views are rank-generic (`ArrayViewMut<'_, T, N>`); `fftshift_axes` already
  takes `ArrayView<'_, T, N>` and `axes: &[usize]` and returns `ApolloResult`.
- None of the 78 `pub fn` in `api/` or the 49 `forward_*`/`inverse_*` methods
  of the 1-D, 2-D and 3-D plans returns a `Result`; inputs are checked by
  `Shape*::new(..).expect`, `assert_eq!` and `debug_assert_eq!`.
  [ADR 0072](0072-fallible-public-fft-api.md) (Proposed) makes them fallible.
- Nine plan methods carry a mode in their names (`inverse_*_unnorm_*`), called
  from apollo-nufft, apollo-qft and apollo-sht.
- The one retained-bytes measure, `batched::cache::retained_bytes_f64`, is
  `#[cfg(test)] pub(crate)` and counts batched plans and folds alone.

NumPy, `numpy.fft` reference, section "Normalization" (v2.5 manual, read
2026-11-24): `"backward"` leaves the forward unscaled and scales the inverse
by 1/n; `"ortho"` scales both by 1/sqrt(n); `"forward"` scales the forward by
1/n and leaves the inverse unscaled. `numpy.fft.fftn` transforms a repeated
axis again and raises `IndexError` for an axis outside the array; NumPy 2.5.2,
run this session, raises `ValueError` for an empty input.

Version class. The checklist and `gap_audit.md` call 0.32.0 [minor], yet the
ladder records 0.30.0 as [major], ADR 0067 calls slice 2 "breaking: a minor
version bump before 1.0" and the board tags `APOLLO-CAP-NORMALIZATION-MODES`
[major] here: a 0.x minor bump is the breaking boundary (Cargo's pre-1.0 rule,
recalled; its page was unreachable). Extending the exhaustive `Normalization`
already breaks.

## Decision

1. **Mode.** `Normalization` becomes `#[non_exhaustive]` with `Backward`
   (`FftwCompatible` renamed, the old name deleted), `Orthonormal` and
   `Forward`, passed as a value, `norm: Normalization`. One `match` at the
   operation boundary selects the monomorphized `Direction` passes (`Forward`,
   `Inverse`, `InverseUnnormalized`) and at most one scale pass, its factor
   computed once in `T`. With N the product of the transformed extents, the
   forward scales by 1, 1/sqrt(N), 1/N and the inverse by 1/N, 1/sqrt(N), 1;
   under `Backward` the plans run today's passes, fused 1/n included, so their
   outputs stay bitwise. The axis-set functions (2) and every plan
   `forward_*`/`inverse_*` take the mode; the plans gain `norm` in the
   signature change of (3), so their callers migrate once. The nine
   `inverse_*_unnorm_*` plan methods are deleted: their callers pass
   `Forward`, and QFT passes `Orthonormal` in place of its own 1/sqrt(N). The
   rank-named `api` functions take no mode, keep `Backward` (NumPy's default)
   and call the same plan methods: no kernel is duplicated.
2. **Axis sets.** Rank-generic and in place over Leto views of any layout,
   with `T` bounded as the complex `api` functions are (`f64`, `f32`, `F16`):
   ```rust
   pub fn fft_axes<T, const N: usize>(field: ArrayViewMut<'_, Complex<T>, N>,
       axes: &[usize], norm: Normalization) -> ApolloResult<()>;
   pub fn ifft_axes<T, const N: usize>(field: ArrayViewMut<'_, Complex<T>, N>,
       axes: &[usize], norm: Normalization) -> ApolloResult<()>;
   ```
   `&[k]` is `numpy.fft.fft(a, axis=k)` and a set is `fftn(a, axes=...)`; an
   empty set leaves the field unchanged, as in `fftshift_axes`. A view that is
   not C-dense stages through `with_c_order_view` in the 2-D scratch role at
   every rank, disjoint from the 3-D X and Y roles of the passes. For each
   axis `a` the C-order shape collapses to `[outer, n_a, inner]` (the extent
   products before and after `a`), and the existing 3-D pass runs the lane
   closure the 3-D plan builds (`lane_over`, with the bounded `FftPlan1D`
   owner for four-step lengths) over `n_a`: `axis2` when `inner = 1`, else
   `axis0` when `outer = 1`, else `axis1`; an extent of 1 is skipped. No
   kernel and no N-D plan type is added; the eight plan-level axis methods are
   deleted. Real forms, delivered with the Python surface of
   `M-backend-and-python-reach`: `rfft_axis(field: ArrayView<'_, T, N>, axis,
   norm)` returns `ApolloResult<Array<Complex<T::PlanScalar>,
   MnemosyneStorage<_>, N>>` with `n/2 + 1` bins along `axis`, and
   `irfft_axis(spectrum, axis, len: Shape1D, norm)` takes the output length as
   `numpy.fft.irfft(n=...)` does. NumPy's crop and pad (`n`, `s`) are out of
   scope. Leto views carry every form; apollo-leto-interop adds nothing.
3. **Fallibility.** ADR 0072 applies in place, with no `try_` family, to the
   78 `api` functions and to the 32 plan execute methods left once (1) and (2)
   delete 17 of the 49; `fft_axes` and `ifft_axes` are fallible from birth.
   `ApolloError` becomes `#[non_exhaustive]` in the same change and gains no
   variant. Preconditions map to:
   - zero-length input or transformed extent: `Validation` naming the field
     (`n`, `nx`, `ny`, `nz`, `axes`) with value `0`, as `Shape*::new` does;
   - an axis not below the rank, or repeated: `Validation` for `axes`, as
     `fftshift_axes` returns (NumPy repeats the transform; Apollo refuses);
   - an `_into` output, a plan shape against the data, a static `N` against
     the runtime length, and an `irfft_axis` extent against `len / 2 + 1`:
     `ShapeMismatch { expected, actual }`;
   - a buffer a route needs contiguous: `NonContiguous`; kernel-internal
     invariants stay `expect` calls stating the invariant.
4. **Retained bytes.** `pub fn retained_plan_bytes() -> usize`, a crate-root
   function like `clear_plan_caches()`, counts over every scalar each distinct
   table reachable from an entry of the twiddle, four-step, Rader, PFA,
   Bluestein, resident, batched or plan caches (a live `Weak` or an `Arc`),
   once by address as `len * size_of::<element>()`, plus `size_of` of each
   plan-cache and coprime-factor entry. It reads the process-wide caches and
   the calling thread's local ones; another thread's ring shows only through
   the tables it keeps alive. `clear_plan_caches()` also prunes dead `Weak`
   entries and empties the coprime-factor map and the calling thread's local
   tables, so with no plan held the value returns to its post-clear baseline.
   The fixed statics (the 4,096-entry factor array, the Rader generator table)
   are excluded: their size is independent of the workload. The count presumes
   [ADR 0068](0068-bounded-plan-and-table-caches.md) slice 2.

## Rejected alternatives

- **Additive 0.32.0**: a second mode type beside the one-variant enum, and
  fallible siblings beside the panicking forms. It contradicts Accepted ADR
  0067, splits one concept over two types, keeps two public paths per
  operation and 127 panicking entry points, and moves the break to 0.33.0. A
  redirect to it adds `NormMode` for (2) alone, keeps the plan signatures,
  `_unnorm_` methods and panics, and revises ADR 0067 slice 2.
- **A type-parameter mode**: bindings choose `norm` at run time; a value costs
  one branch per call, not the per-lane flag ADR 0067 rejected. **`norm` on
  the rank-named functions**: a second migration of every caller for a mode
  `fft_axes` gives at every rank.
- **An `FftPlanNd`**: its state is per-axis `FftPlan1D`s the bounded plan
  cache owns. **Keeping the plan axis methods**: a second, rank-limited axis
  API.
- **`retained_plan_bytes::<T>()`**: factor, PFA and Rader-order tables have no
  scalar, and the clear it pairs with spans every scalar.

## Consequences

- 0.32.0 breaks the renamed variant, `#[non_exhaustive]` on `Normalization`
  and `ApolloError`, every `api` and plan execute signature, and the deleted
  `_unnorm_` and axis methods; `BREAKING CHANGE:` footers carry the migration.
  `BackendCapabilities::normalization` stays `Backward` until ADR 0071 lands.
- The rank-named family overlaps `fft_axes` for whole complex arrays at ranks
  1-3 and keeps the static-`N`, half-spectrum and slice forms that `fft_axes`
  lacks; `M-stable-transform-suite` decides whether 1.0 keeps both.
- `ApolloError` payloads stay `String`; typing them is a later break. ADR
  0076's interop "axis views" row takes a revision note on acceptance.

## Overturning evidence

- A user veto of the 0.32.0 break: the additive redirect above.
- Axis plans needed outside the bounded cache: an `FftPlanNd`. A scale pass
  above the `Backward` drift band: fuse it per route. NumPy-parity cases
  needing repeated axes or crop and pad: widen (2).
