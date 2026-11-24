# ADR 0072: Fallible public FFT API

Status: Proposed  \
Date: 2026-10-10  \
Change class: [major]  \
Board item: `APOLLO-CAP-FALLIBLE-API` (audit item 7)  \
Revised: 2026-11-24, counts re-measured at `b5e8b70b` and two claims corrected (silent output, Python checks) because the PR 731 judges found them false in `api/irfft/dimension_1d.rs:124` and `bindings/support.rs:136`; [ADR 0077](0077-numpy-fft-semantics-surface.md) deletes 17 of the 49 plan methods, gives the rest `norm`, adds the `RealFftData` methods and fixes the error map and the 0.32.0 release

## Context

The public FFT API panics on bad input instead of returning errors,
measured in release at audit time and re-verified 2026-10-10:
`fft_1d_slice(&[])` panics on `Shape1D::new(n).expect(..)`,
`fft_1d_complex_into` with a length mismatch panics inside Leto
`assign` (the API's own check is `debug_assert_eq!`, absent in
release), and the static-`N` entry panics on a length/`N` mismatch.
36 lines of `api/*.rs` carry `debug_assert_eq!` shape checks at
`b5e8b70b`, and `Shape*::new(..).expect(..)` appears at nearly every
top-level entry. A missing release check usually ends in a panic where a
service or binding needs an error, but not always: `ifft_1d_array_into`
(`api/irfft/dimension_1d.rs:124`) with `field_hat` of length 10, `out` 8 and
`scratch` 10 runs the real split over bins 0 to 4 and returns eight wrong
reals without panicking, and `_into_spectrum_scratch` is silent in release
(its only check is the `debug_assert_eq!` at :102). The 2-D and 3-D
analogues were not read; `APOLLO-COR-IRFFT-SPLIT-LENGTH` fixes the two
forms above and classifies the other 35 sites.

Two facts decide the shape of the fix. First, the acceptance oracle
requires the API to *return* the specific `ApolloError` variant with
the offending values, in release, over a decision table of
empty/length-mismatch/shape-mismatch/static-`N`-mismatch by
dimensionality, domain, and in-place form. Documenting `# Panics`
cannot satisfy an oracle written in terms of returned errors.
Second, the plan types are public API themselves (`FftPlan1D/2D/3D`
and the static plans are re-exported from the crate root), so
validating only at the `api` layer would leave the plan execute
methods — 49 `forward_*`/`inverse_*` functions taking arbitrary
slices and views — panicking on the same mismatches for downstream
plan holders.

## Decision

Make the fallibility total at the public boundary, with no
`try_`-prefixed parallel API:

- The 78 public `api` functions become fallible, returning
  `ApolloResult`, with `Shape*::new(..).expect(..)` replaced by `?`
  (the constructors already return `ApolloResult`) and every
  `debug_assert_eq!` length/shape check replaced by a release-checked
  typed error carrying the expected and actual values. The
  static-`N` entries check the runtime length against `N` the same
  way.
- The 32 public plan-level `forward_*`/`inverse_*` execute methods
  that ADR 0077 keeps, and the `RealFftData` execute methods, return
  `ApolloResult`, validating buffer length (and view shape) against the plan
  before dispatching. Kernel callers inside the crate build the plan from the
  data's own length, so a mismatch is unreachable there; they call a
  `pub(crate)` core that the checked public method wraps, per ADR 0077
  decision 3, and no `expect` shim remains on a migrated public path.
- Kernel-internal invariants stay panics: the Leto
  contiguity/spectrum-shape expects inside the api bodies assert
  postconditions of Apollo's own construction, not input
  validation, and are out of scope per the audit's non-goals.
- Valid inputs are bitwise unchanged: the checks precede dispatch
  and touch nothing on the success path.

This is [major]: every fallible signature changes, and downstream
callers (including the Python bindings, which check only C-contiguity and
precision today, so an empty array reaches the panicking api) update in
the same release.

## Alternatives rejected

- Document `# Panics` consistently and keep infallible signatures
  ([minor]): rejected because it fails the acceptance oracle, which
  requires returned errors, not documented panics.
- `try_`-prefixed fallible twins alongside the panicking originals:
  rejected because parallel APIs double the public surface and
  leave the panicking half as the default every example reaches
  for; the audit scope prescribes the `try_`-free design.
- Boundary-only validation (fallible `api`, infallible plans):
  rejected because the plans are public API with the same
  length-taking execute methods; downstream plan holders would keep
  the panic the item exists to remove.

## Consequences

The executing change migrates all 78 `api` functions, the 32 plan
execute methods, and their in-repo callers (kernels composing plans,
other transform crates, benches, tests, doctests) in dependency-ordered
slices, each compiling with its callers migrated,
with the audit's decision table committed as a release-mode test
asserting the exact `ApolloError` variant and values per cell, plus
a bitwise-equality guard on valid inputs.

## Verification

The decision-table test passes in release; the full workspace gate
(including doctests and the Python parity suites that exercise the
previously panicking inputs) is green; no `debug_assert` remains in
`api/`, and no `.expect(` remains on a migrated input-validation
path.
