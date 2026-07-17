# ADR 0031: Validation suite concern tree

- Status: Accepted
- Date: 2026-07-16
- Change class: internal architecture refactor

## Context

`apollo-validation` keeps a 974-line `application::suite::mod` file despite
already having distinct suite responsibilities: orchestration, CPU and GPU FFT
laws, NUFFT checks, external comparisons, published-reference fixtures,
benchmarks, environment reporting, and test contracts. The file exceeds the
workspace leaf-size target and makes the module manifest own implementation
concerns.

## Decision

Partition the existing implementations into one private leaf per concern while
retaining `application::suite` as the only public API path. The module manifest
owns declarations, shared private limits, and curated re-exports; it contains no
suite computation. The stale hardcoded GPU availability field/call is removed
under ADR 0013 during the same takeover so the validation report records only
typed acquisition outcomes. Numerical formulas, fixtures, tolerances, and
result construction remain unchanged.

## Invariants and verification

For each public suite function `f`, the refactor preserves its numerical value
contract: `f_before(input) = f_after(input)` for transform outputs, analytical
errors, and propagated provider failures. The obsolete availability report field
is intentionally absent; Hephaestus acquisition is the sole capability
evidence. The existing validation contracts provide the empirical oracle: FFT
round-trip and Parseval checks, CPU/GPU differentials, NUFFT adjoint and
relative-error bounds, published-reference values, and benchmark report schema.
No new tolerance, fallback, or provider wrapper is introduced.

## Consequences

- Each concern has one canonical leaf below `application::suite`.
- Existing public callers retain `apollo_validation::application::suite::*`.
- The refactor preserves value-semantic tests and warning-clean API docs; the
  validation package passes 10/10 Nextest tests, all-targets Clippy, and
  rustdoc.
