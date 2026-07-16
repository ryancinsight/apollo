# ADR 0025: Root verification boundaries

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking verification-boundary cleanup

## Context

Ten transform crates publish a crate-root `verification` module whose sole
contents compile under `cfg(test)`. The release build therefore exports empty
test paths. The DCT/DST root additionally holds 672 lines of unrelated
one-dimensional, multidimensional, direct-law, and property contracts in one
module.

## Decision

Gate each root verification module with `cfg(test)` and make it private. Do not
provide a compatibility alias. Split the DCT/DST module into concern-named
private leaves for plan execution, multidimensional execution, direct transform
laws, and property laws. The existing test values, CPU oracles, fixtures, and
derived finite-precision tolerances stay byte-for-byte unchanged within their
respective test bodies.

## Theorem and evidence boundary

This restructuring changes no transform formula or theorem. In particular,
the DCT/DST inverse-pair and self-inverse laws retain their existing analytical
specifications and property-test evidence. Compilation, value-semantic tests,
and API-surface scans provide empirical and release-boundary evidence only;
they are not machine-checked proofs of accelerator behavior.

## Consequences

- The affected crates expose no public root `verification` path.
- Test ownership remains transform-local and the DCT/DST tree has one
  concern per leaf below the 500-line target.
- Apollo retains no wrapper around test evidence, Leto remains the host-array
  boundary, and Hephaestus retains typed device mechanics.
