# ADR 0078: Release sequence and consumers of the 0.32.0 break

- Status: Proposed
- Date: 2026-11-24; [major], `apollo-fft` 0.32.0; basis: apollo `origin/main`
  at the base of PR 731
- Items: `APOLLO-CAP-BATCH-AXIS-ND`, `APOLLO-CAP-NORMALIZATION-MODES`,
  `APOLLO-CAP-FALLIBLE-API`, `APOLLO-CAP-KERNEL-SURFACE`

## Context

[ADR 0077](0077-numpy-fft-semantics-surface.md) breaks the public surface of
`apollo-fft` over four items, so the manifest version, the sibling
requirements, the order in which consumers migrate and the Atlas development
overlay need one sequence. The overlay patches only the git source of
`apollo-fft` (`.cargo/config.toml`), and the Cargo Book (overriding
dependencies, read 2026-11-24) says the patched version "will affect whether
the patch is used".

## Decision

1. **Manifest.** The first breaking slice to land
   (`APOLLO-CAP-BATCH-AXIS-ND`) sets `apollo-fft` 0.32.0 and moves the 18
   sibling `version = "0.31.0"` requirements (`apollo-python` is path-only) in
   the same commit; publication needs release authority. Git-sourced
   consumers keep 0.31.0 until each migrates once, against the finished
   surface.
2. **Reverse dependencies** at the fetched defaults:
   - direct `api` calls: kwavers 90 in 42 files (plus 14 plan calls, 2 axis
     calls, 4 `RealFftData` calls, `FftwCompatible` once), CFDrs 8 in 3,
     coeus 4 in 1, helios 3 in 1;
   - requirement edge: ritk, with 10 `kernel::fft_*` calls in 5 files;
   - transitive: asclepius, through the `coeus-autograd` lock;
   - unaffected: the other 21 members (`consus` names only
     `apollo-dctdst-core`); none has an `apollo-fft`/`-nufft`/`-sht` edge.
3. **Order.** Siblings, coeus, ritk and asclepius, then helios, kwavers and
   CFDrs. kwavers re-exports `Normalization` and `FftPlan1D/2D/3D`
   (`kwavers-math/src/fft/mod.rs:18-21`) and exposes
   `FourierBackend::normalization()`, so its published crates break too.
   Siblings returning `ApolloResult` (apollo-nufft, apollo-sft) or implementing
   `GpuTransformPlanner` (apollo-sht) version by `cargo-semver-checks` class,
   and the CFDrs and ritk requirements follow.
4. **Overlay window.** Derived from reading `scripts/atlas-stack-overlay.py`
   on Atlas `origin/main` fetched 2026-11-24 (`satisfies`, `build_overlay`,
   `check`, `check_locks`); the check was not run. From slice 1 until
   `COEUS-APOLLO-NUMPY-FFT` lands (after the third breaking slice,
   `APOLLO-CAP-FALLIBLE-API`, at the earliest), coeus
   (`Cargo.toml:47`, git+version `0.31.0`, no `rev`) fails `satisfies`, so the
   generator drops the `apollo-fft` patch and `check` exits 1 with
   REQUIREMENT LAG (coeus), PIN DRIFT (the coeus and asclepius locks) and a
   stale overlay. The `atlas-stack-overlay` workflow, not a required check and
   already red on Atlas main, stays red for the window. Owner: the upstream
   integrator, through the Atlas epic `ATLAS-APOLLO-NUMPY-FFT-SURFACE`, which
   records the window, takes the coeus landing as its re-open trigger and
   regenerates the overlay then. Hypothesis, not run: a coeus lock
   re-resolved without its git pin finds no `^0.31.0` candidate on apollo's
   default branch (0.32.0 after slice 1), so the gate's printed cure,
   `cargo update` in coeus, would fail there; the generator's docstrings
   predict a resolver failure even with the pin, which this ADR leaves
   unresolved.

## Rejected alternatives

- **Bumping the manifest in the last slice**: overlay consumers meet each
  break as it lands.

## Consequences

- While the window is open `atlas-stack-overlay` exits 1 for this cause as
  well as the lags it already reports; the window and its trigger are
  recorded in the epic rather than the gate silenced.
- The Atlas epic and the six member items are to be filed in the Atlas filing
  sweep; none exists on the Atlas or the six member boards at this basis.

## Overturning evidence

`cargo metadata` in a coeus export after slice 1, once without its lock and
once with the committed lock under the committed overlay: a lockless resolve
that succeeds falsifies the hypothesis; a failure with the committed lock
confirms the generator's docstring and reopens the handling (coeus
advancing its requirement at slice 1 is the alternative). A user veto of the
0.32.0 break (ADR 0077) removes the window.
