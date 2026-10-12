# ADR 0072: Fallible public FFT API

Status: Proposed  \
Date: 2026-10-10  \
Change class: [major]  \
Board item: `APOLLO-CAP-FALLIBLE-API` (audit item 7)

## Context

The public FFT API panics on bad input instead of returning errors,
measured in release at audit time and re-verified 2026-10-10:
`fft_1d_slice(&[])` panics on `Shape1D::new(n).expect(..)`,
`fft_1d_complex_into` with a length mismatch panics inside Leto
`assign` (the API's own check is `debug_assert_eq!`, absent in
release), and the static-`N` entry panics on a length/`N` mismatch.
24 `debug_assert_eq!` shape checks remain in `api/*.rs` (down from 36
at audit), and `Shape*::new(..).expect(..)` appears at nearly every
top-level entry. Nothing corrupts output silently — the failure mode
is a panic where a service or binding needs an error.

Two facts decide the shape of the fix. First, the acceptance oracle
requires the API to *return* the specific `ApolloError` variant with
the offending values, in release, over a decision table of
empty/length-mismatch/shape-mismatch/static-`N`-mismatch by
dimensionality, domain, and in-place form. Documenting `# Panics`
cannot satisfy an oracle written in terms of returned errors.
Second, the plan types are public API themselves (`FftPlan1D/2D/3D`
and the static plans are re-exported from the crate root), so
validating only at the `api` layer would leave the plan execute
methods — 41 `forward_*`/`inverse_*` functions taking arbitrary
slices and views — panicking on the same mismatches for downstream
plan holders.

## Decision

Make the fallibility total at the public boundary, with no
`try_`-prefixed parallel API:

- The 59 public `api` functions become fallible, returning
  `ApolloResult`, with `Shape*::new(..).expect(..)` replaced by `?`
  (the constructors already return `ApolloResult`) and every
  `debug_assert_eq!` length/shape check replaced by a release-checked
  typed error carrying the expected and actual values. The
  static-`N` entries check the runtime length against `N` the same
  way.
- The 41 public plan-level `forward_*`/`inverse_*` execute methods
  return `ApolloResult`, validating buffer length (and view shape)
  against the plan before dispatching. Internal kernel callers,
  which hold correctly-sized buffers by construction, propagate
  with `?`; no `expect` shims remain on migrated paths.
- Kernel-internal invariants stay panics: the Leto
  contiguity/spectrum-shape expects inside the api bodies assert
  postconditions of Apollo's own construction, not input
  validation, and are out of scope per the audit's non-goals.
- Valid inputs are bitwise unchanged: the checks precede dispatch
  and touch nothing on the success path.

This is [major]: every fallible signature changes, and downstream
callers (including the Python bindings, which currently re-validate
everything themselves and can then drop that duplication) update in
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

The executing change migrates all 59 `api` functions, all 41 plan
execute methods, and their in-repo callers (kernels composing plans,
other transform crates, benches, tests, doctests) in one commit,
with the audit's decision table committed as a release-mode test
asserting the exact `ApolloError` variant and values per cell, plus
a bitwise-equality guard on valid inputs.

## Verification

The decision-table test passes in release; the full workspace gate
(including doctests and the Python parity suites that exercise the
previously panicking inputs) is green; no `debug_assert` remains in
`api/`, and no `.expect(` remains on a migrated input-validation
path.
