# ADR 0044: Shape descriptors validate at construction

- Status: Proposed (2026-09-02)
- Item: `ATLAS-APOLLO-SHAPE1D-PRIVACY`
- Driving evidence: `crates/apollo-fft/src/domain/metadata/shape.rs` declares
  `pub n` / `pub nx, ny` / `pub nx, ny, nz` on `Shape1D`/`Shape2D`/`Shape3D`
  beside validating constructors (`new` rejects zero lengths); 42 in-tree
  `Shape1D { n }` literals and two in kwavers (`crates/kwavers-math/src/fft/mod.rs`) construct
  shapes without that validation (2026-09-02 sweep).

## Context

The shape descriptors are validating newtypes in intent: `Shape1D::new(n)`
refuses `n == 0` so every plan is built on a non-empty length. Their fields
are `pub`, so a struct literal is a second construction path that skips the
check, and consumers use it — 42 sites in this workspace, mostly tests, and
two in kwavers. A zero length reaches plan construction through that path and
fails later, or not at all, with a message about the plan rather than the
shape. The same holds for `Shape2D` and `Shape3D`, whose per-axis lengths
carry the same invariant.

The standards call this primitive-obsession-by-exposure: a validity boundary
that is not the only way in is not a boundary (validation is
privacy-enforced; `#[non_exhaustive]` forecloses cross-crate literals).

## Decision

1. `Shape1D`, `Shape2D` and `Shape3D` make their length fields private and
   gain `#[must_use] pub const fn` accessors — `n()`, `nx()`/`ny()`,
   `nx()`/`ny()`/`nz()` — plus `len()` on the 1-D shape where the total
   element count reads better. The validating `new` constructors are the only
   construction path; the structs carry `#[non_exhaustive]`.
2. Every in-tree literal migrates in the same change: `Shape1D { n }` becomes
   `Shape1D::new(n)?` in fallible code and
   `Shape1D::new(n).expect("invariant: <why n > 0 here>")` where the length is
   a compile-time or test constant. No forwarding constructor, alias, or
   `From<usize>` bridge is kept.
3. kwavers, the one external consumer, migrates its two sites in the
   co-evolution unit that advances its apollo pin.

Change class: **[major]** — removing `pub` fields is a public-contract break
(`cargo-semver-checks`: `struct_pub_field_missing`). It ships with the
other Unreleased breaks `apollo-fft` already carries at 0.27.0.

## Alternatives

- **Keep `pub` fields, document the invariant.** Rejected: documentation is
  not a boundary; the literal path stays open to every consumer.
- **`pub(crate)` fields.** Rejected: in-crate literals are the majority of the
  42 sites and bypass the check just the same.
- **A `From<usize>` impl for ergonomics.** Rejected: it is `new` without the
  validation, the same bypass under a different name (a fallible `From` is a
  mislabelled `TryFrom`).

## Verification

- `cargo-semver-checks` on the change reports exactly the field removals
  (the declared class) and nothing else.
- `Shape*::new` rejection tests stay; a `compile_fail` doctest shows the
  literal no longer compiles outside the crate.
- The workspace gate is green with every literal migrated; kwavers builds
  against the advanced pin.

## Migration

`shape.n` → `shape.n()` (and `nx`/`ny`/`nz`); `Shape1D { n }` →
`Shape1D::new(n)?` or `.expect(...)`. The CHANGELOG `Breaking` entry names
the accessors and the constructor.
