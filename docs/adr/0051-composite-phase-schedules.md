# ADR 0051: Composite phase schedules

- **Status:** Accepted
- **Date:** 2026-09-08
- **Class:** [major] [arch]
- **Item:** [APOLLO-CODELET-SCHEDULE-CONTROLS](../../backlog.md#apollo-codelet-schedule-controls)

## Decision

Composite codelets use one generated arithmetic body. The measured 50-point
Good–Thomas and 144-point Cooley–Tukey codelets admit a private compile-time
schedule parameter. `Fused` inlines each phase closure; test-only `Split`
places each closure behind a non-inlined call. Each closure monomorphizes
over the same scalar and direction as its enclosing codelet. Production
`ShortDft` implementations select `Fused` explicitly.

Existing macro pair inputs retain their generated function signatures and
inline phases. Scheduling is opt-in per pair through `scheduled_pairs`, and
the macro accepts only the two measured factorisations there — Good-Thomas
`(2, 25)` and Cooley-Tukey `(12, 12)` — rejecting any other pair at
expansion. Admitting a third length is a macro-crate edit, not a caller
choice. This adds a macro capability without
changing ordinary codelet call signatures. The test-only phase helpers are
removed as described below; this is a breaking macro expansion change.

The safe schedule trait is sealed to these two implementations and exposes
only `SPLIT_PHASES`. Generated code owns the arithmetic closure and invokes
it synchronously exactly once, either directly or through its own generic
non-inlined function. A schedule never receives the closure. Both branches
propagate panics, so normal return implies scratch initialization completed.
Constant propagation removes the unselected boundary per specialization.

The outer codelet owns one uninitialized scratch array. Scheduling changes
the phase call boundaries only: initialization, arithmetic order, twiddle
expressions, permutations and scratch storage remain the generated body's
responsibility. Other composite sizes and fixed Good–Thomas dispatch omit
the schedule parameter. No scalar policy is added.

## Alternatives and evidence

The inherited routing work duplicates the fused family so its benchmark
can force a control independently of production selection. A schedule
parameter supplies that control without cloned bodies, unused exports or
dead-code allowances. Runtime scalar selection adds a policy to arithmetic
traits despite the absence of accepted production routing evidence.

Automatically scheduling the two existing pair inputs was rejected during
review: it changes generated generic arity for callers that never asked for
scheduling. The consumer-root trait path is not a discriminator — the
accepted design emits the same path whenever the opt-in is used — so source
compatibility for unscheduled pairs is the whole of the argument. Existing
Apollo-specific paths do not exempt an exported macro from it. Explicit opt-in retains the old
expansion without a forwarding wrapper or a second arithmetic body.

A callback-bearing schedule trait was rejected as unsound at the exported
macro boundary. Its consumer-root path can resolve to a safe trait with a
no-op implementation, regardless of an unsafe-trait requirement in Rustdoc.
That skips column writes before the generated `assume_init_mut`. A bound
does not encode trait unsafeness. Restricting the policy to a constant and
retaining execution inside generated code removes that trust dependency.

## Migration

Consumers calling generated `dftN_impl::<F, INVERSE>` functions need no
change unless they opt into `scheduled_pairs: [(2, 25), (12, 12)]`, listing
only pairs present in that invocation. Existing pair-list inputs continue
to emit that signature and do not require a schedule trait.

The former test-only `dft50_rows`, `dft50_cols`, `dft144_rows` and
`dft144_cols` functions are removed. Tests using them select the scheduled
codelet and compare complete transforms under `Fused` and `Split`; the
codelet owns and initializes its scratch. Consumers must not recreate the
removed phase bodies or retain a separate scratch-initialization path.
Apollo's tests and measurement controls migrate in the same change.

The previous macro documentation omits these helpers, but their generated
`pub(crate)` visibility permits use inside a consuming crate's tests.
Removing them therefore receives a breaking classification rather than an
assumption that no external consumer uses them. This development change
does not authorize a package release or version bump.

The [PR 346](https://github.com/ryancinsight/apollo/pull/346) table is a
forward-only measurement on one local machine. It does not establish inverse,
cross-machine, whole-transform or artifact-size acceptance. Production
remains fused. A future promotion
requires controlled measurements for its intended machines and workloads,
plus the same behavioral and size gates; this decision makes no speed claim.

## Verification

The generic leaf test compares selected, forced-fused and forced-split
outputs exactly at lengths 50 and 144, both supported precisions and both
directions. Each case copies one sinusoidal input so transcendental
evaluation cannot differ between controls. Exact equality is justified by
unchanged arithmetic order, not by a relaxed error tolerance. Existing
independent transform tests still provide the mathematical oracle.

The probe retains its source values, batches, case order, timed closures
and measurement configuration. Its control wrappers call the generated
codelet; scratch allocation and phase ordering no longer live in the probe.
Generated code and linked-image comparison must establish that the fused
schedule adds no scratch work or attributed size regression. Formatting
and source comparison alone cannot establish those compiler properties.
